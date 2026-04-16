from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from typing import Any

import joblib
import pandas as pd
from django.conf import settings
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_pipeline import load_booking_dataset, load_reference_tables


ARTIFACT_FILE_NAME = 'commission_forecast_bundle.joblib'
PICKLE_ARTIFACT_FILE_NAME = 'commission_forecast_bundle.pkl'
METRICS_FILE_NAME = 'commission_forecast_metrics.json'

FEATURE_COLUMNS = [
    'Consultation_fees',
    'Age',
    'Gender',
    'payment_method',
    'department',
    'Experience_years',
    'doctor_type',
    'Rating_avg',
    'Rating_count',
    'city',
    'Hospital_type',
    'Emergency_service',
    'Service_charge',
    'booking_month',
    'booking_day',
    'booking_weekday',
    'booking_week_of_year',
]

NUMERIC_FEATURES = [
    'Consultation_fees',
    'Age',
    'Experience_years',
    'Rating_avg',
    'Rating_count',
    'Service_charge',
    'booking_month',
    'booking_day',
    'booking_week_of_year',
]

CATEGORICAL_FEATURES = [column for column in FEATURE_COLUMNS if column not in NUMERIC_FEATURES]


@dataclass
class ForecastResult:
    summary: dict[str, Any]
    breakdowns: dict[str, list[dict[str, Any]]]
    model_metrics: dict[str, Any]
    training_summary: dict[str, Any]


def artifact_dir() -> Path:
    path = Path(settings.FORECAST_ARTIFACT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_path() -> Path:
    return artifact_dir() / ARTIFACT_FILE_NAME


def pickle_artifact_path() -> Path:
    return artifact_dir() / PICKLE_ARTIFACT_FILE_NAME


def metrics_path() -> Path:
    return artifact_dir() / METRICS_FILE_NAME


def build_training_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                'num',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                'cat',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OneHotEncoder(handle_unknown='ignore')),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
    )

    return Pipeline(
        steps=[
            ('preprocessor', preprocessing),
            ('model', model),
        ]
    )


def _serialize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str)):
            serialized[key] = value
        else:
            serialized[key] = float(value)
    return serialized


def train_and_persist_model_bundle(force_retrain: bool = False) -> dict[str, Any]:
    if artifact_path().exists() and not force_retrain:
        bundle = joblib.load(artifact_path())
        if not pickle_artifact_path().exists():
            with pickle_artifact_path().open('wb') as pickle_file:
                pickle.dump(bundle, pickle_file)
        return bundle

    dataset = load_booking_dataset()
    target = dataset['commission_amount']
    features = dataset[FEATURE_COLUMNS]

    train_size = max(1, int(len(dataset) * 0.8))
    X_train = features.iloc[:train_size]
    X_test = features.iloc[train_size:]
    y_train = target.iloc[:train_size]
    y_test = target.iloc[train_size:]

    pipeline = build_training_pipeline()
    pipeline.fit(X_train, y_train)

    baseline_prediction = [float(y_train.mean())] * len(y_test) if len(y_test) else []
    predictions = pipeline.predict(X_test) if len(y_test) else []

    metrics = {
        'model_name': 'RandomForestRegressor',
        'training_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
        'mae': round(float(mean_absolute_error(y_test, predictions)), 2) if len(y_test) else 0.0,
        'rmse': round(float(mean_squared_error(y_test, predictions) ** 0.5), 2) if len(y_test) else 0.0,
        'mape': round(
            float(((y_test.reset_index(drop=True) - pd.Series(predictions)).abs() / y_test.reset_index(drop=True)).mean() * 100),
            2,
        )
        if len(y_test)
        else 0.0,
        'r2': round(float(r2_score(y_test, predictions)), 3) if len(y_test) else 0.0,
        'baseline_mae': round(float(mean_absolute_error(y_test, baseline_prediction)), 2) if len(y_test) else 0.0,
        'baseline_rmse': round(float(mean_squared_error(y_test, baseline_prediction) ** 0.5), 2) if len(y_test) else 0.0,
        'baseline_mape': round(
            float(((y_test.reset_index(drop=True) - pd.Series(baseline_prediction)).abs() / y_test.reset_index(drop=True)).mean() * 100),
            2,
        )
        if len(y_test)
        else 0.0,
    }

    training_summary = {
        'source_file': ', '.join(
            [
                settings.FORECAST_DOCTOR_FILE.name,
                settings.FORECAST_PATIENT_FILE.name,
                settings.FORECAST_HOSPITAL_FILE.name,
            ]
        ),
        'bookings': int(len(dataset)),
        'date_start': dataset['Payment_date'].min().date().isoformat(),
        'date_end': dataset['Payment_date'].max().date().isoformat(),
        'unique_departments': int(dataset['department'].nunique()),
        'unique_cities': int(dataset['city'].nunique()),
        'average_commission': round(float(dataset['commission_amount'].mean()), 2),
        'average_commission_pct': round(float(dataset['commission_pct'].mean()), 2),
        'average_daily_bookings': round(float(dataset.groupby('Payment_date').size().mean()), 2),
        'cancellation_learning': 'Cancellation rate is not present in the dataset, so it is applied as a scenario assumption.',
    }

    bundle = {
        'pipeline': pipeline,
        'history': dataset,
        'metrics': _serialize_metrics(metrics),
        'training_summary': training_summary,
    }
    joblib.dump(bundle, artifact_path())
    with pickle_artifact_path().open('wb') as pickle_file:
        pickle.dump(bundle, pickle_file)
    metrics_path().write_text(json.dumps(bundle['metrics'], indent=2), encoding='utf-8')
    return bundle


def load_model_bundle() -> dict[str, Any]:
    return train_and_persist_model_bundle(force_retrain=False)


def get_forecast_options() -> dict[str, list[str]]:
    history = load_model_bundle()['history']
    doctor, _, hospital = load_reference_tables()
    doctor_directory = (
        doctor.merge(
            hospital[['Hospital_id', 'Hospital_name', 'District']],
            on='Hospital_id',
            how='left',
        )
        .sort_values(['Doctor_name', 'Doctor_id'])
    )
    consultation_service_combinations = (
        history[['Consultation_fees', 'Service_charge']]
        .dropna()
        .drop_duplicates()
        .sort_values(['Consultation_fees', 'Service_charge'])
    )

    revenue_packages = [
        {
            'id': f"{row.Consultation_fees}|{row.Service_charge}",
            'label': f"Consultation: {row.Consultation_fees} - Service: {row.Service_charge}",
        }
        for row in consultation_service_combinations.itertuples()
    ]

    return {
        'departments': sorted(history['department'].dropna().astype(str).unique().tolist()),
        'cities': sorted(history['city'].dropna().astype(str).unique().tolist()),
        'doctor_types': sorted(history['doctor_type'].dropna().astype(str).unique().tolist()),
        'payment_methods': sorted(history['payment_method'].dropna().astype(str).unique().tolist()),
        'genders': sorted(history['Gender'].dropna().astype(str).unique().tolist()),
        'hospital_types': sorted(history['Hospital_type'].dropna().astype(str).unique().tolist()),
        'emergency_service_options': sorted(history['Emergency_service'].dropna().astype(str).unique().tolist()),
        'consultation_fees': sorted(history['Consultation_fees'].dropna().astype(str).unique().tolist()),
        'service_charges': sorted(history['Service_charge'].dropna().astype(str).unique().tolist()),
        'revenue_packages': [{'id': '', 'label': 'All'}] + revenue_packages,
        'doctors': [
            {
                'id': str(row.Doctor_id),
                'label': f"{row.Doctor_name} | {row.Hospital_name} | {row.District}",
            }
            for row in doctor_directory.itertuples()
            if pd.notna(row.Doctor_id) and pd.notna(row.Doctor_name)
        ],
    }


def _filtered_history(history: pd.DataFrame, filters: dict[str, str] | None) -> pd.DataFrame:
    if not filters:
        return history

    filtered = history.copy()
    column_map = {
        'department': 'department',
        'city': 'city',
        'doctor_type': 'doctor_type',
        'payment_method': 'payment_method',
    }

    for filter_key, column_name in column_map.items():
        filter_value = (filters.get(filter_key) or '').strip()
        if filter_value:
            filtered = filtered[filtered[column_name] == filter_value]

    return filtered if not filtered.empty else history


def _build_future_rows(
    history: pd.DataFrame,
    horizon_days: int,
    booking_growth_pct: float,
    cancellation_rate: float,
    avg_daily_bookings: float | None = None,
    window_days: int = 14,
) -> tuple[pd.DataFrame, int]:
    if history.empty:
        return history.copy(), 0

    last_date = history['Payment_date'].max()
    daily_counts = history.groupby('Payment_date').size().tail(window_days)
    base_daily_bookings = (
        float(avg_daily_bookings)
        if avg_daily_bookings is not None and avg_daily_bookings > 0
        else float(daily_counts.mean()) if not daily_counts.empty else 1.0
    )
    planned_bookings = max(1, round(base_daily_bookings * horizon_days * (1 + booking_growth_pct / 100)))
    completed_bookings = max(0, round(planned_bookings * (1 - cancellation_rate / 100)))

    if completed_bookings == 0:
        return history.iloc[0:0].copy(), planned_bookings

    seed_rows = history.tail(min(window_days, len(history))).reset_index(drop=True)
    future_rows: list[dict[str, Any]] = []
    for index in range(completed_bookings):
        template = seed_rows.iloc[index % len(seed_rows)].copy()
        future_date = last_date + pd.Timedelta(days=(index % horizon_days) + 1)
        template['Payment_date'] = future_date
        template['booking_month'] = int(future_date.month)
        template['booking_day'] = int(future_date.day)
        template['booking_weekday'] = future_date.day_name()
        template['booking_week_of_year'] = int(future_date.isocalendar().week)
        future_rows.append(template.to_dict())

    future_frame = pd.DataFrame(future_rows)
    return future_frame, planned_bookings


def _top_breakdown(frame: pd.DataFrame, column: str, label: str) -> list[dict[str, Any]]:
    grouped = (
        frame.groupby(column, dropna=False)
        .agg(
            projected_commission=('predicted_commission_amount', 'sum'),
            projected_bookings=('predicted_commission_amount', 'size'),
        )
        .reset_index()
        .rename(columns={column: label})
        .sort_values('projected_commission', ascending=False)
    )
    records = grouped.to_dict(orient='records')
    normalized: list[dict[str, Any]] = []
    for record in records:
        normalized.append(
            {
                label: record[label] if record[label] == record[label] else 'Unknown',
                'projected_commission': round(float(record['projected_commission']), 2),
                'projected_bookings': int(record['projected_bookings']),
            }
        )
    return normalized


def generate_forecast(
    horizon_days: int = 30,
    booking_growth_pct: float = 0.0,
    cancellation_rate: float = 0.0,
    avg_daily_bookings: float | None = None,
    filters: dict[str, str] | None = None,
) -> ForecastResult:
    bundle = load_model_bundle()
    history: pd.DataFrame = bundle['history']
    pipeline: Pipeline = bundle['pipeline']
    scoped_history = _filtered_history(history, filters)

    future_rows, planned_bookings = _build_future_rows(
        history=scoped_history,
        horizon_days=horizon_days,
        booking_growth_pct=booking_growth_pct,
        cancellation_rate=cancellation_rate,
        avg_daily_bookings=avg_daily_bookings,
    )
    baseline_rows, baseline_planned_bookings = _build_future_rows(
        history=scoped_history,
        horizon_days=horizon_days,
        booking_growth_pct=0.0,
        cancellation_rate=0.0,
        avg_daily_bookings=avg_daily_bookings,
    )

    if not future_rows.empty:
        future_rows = future_rows.copy()
        future_rows['predicted_commission_amount'] = pipeline.predict(future_rows[FEATURE_COLUMNS])
    else:
        future_rows['predicted_commission_amount'] = []

    if not baseline_rows.empty:
        baseline_rows = baseline_rows.copy()
        baseline_rows['predicted_commission_amount'] = pipeline.predict(baseline_rows[FEATURE_COLUMNS])
    else:
        baseline_rows['predicted_commission_amount'] = []

    projected_commission = round(float(future_rows['predicted_commission_amount'].sum()), 2)
    baseline_commission = round(float(baseline_rows['predicted_commission_amount'].sum()), 2)
    uplift_commission = round(projected_commission - baseline_commission, 2)

    summary = {
        'horizon_days': int(horizon_days),
        'planned_bookings': int(planned_bookings),
        'projected_bookings': int(len(future_rows)),
        'baseline_planned_bookings': int(baseline_planned_bookings),
        'projected_commission': projected_commission,
        'baseline_commission': baseline_commission,
        'uplift_commission': uplift_commission,
        'booking_growth_pct': float(booking_growth_pct),
        'cancellation_rate': float(cancellation_rate),
        'avg_daily_bookings': float(avg_daily_bookings) if avg_daily_bookings is not None else None,
        'filters': filters or {},
    }

    breakdowns = {
        'department': _top_breakdown(future_rows, 'department', 'department'),
        'city': _top_breakdown(future_rows, 'city', 'city'),
        'doctor_type': _top_breakdown(future_rows, 'doctor_type', 'doctor_type'),
        'payment_method': _top_breakdown(future_rows, 'payment_method', 'payment_method'),
    }

    return ForecastResult(
        summary=summary,
        breakdowns=breakdowns,
        model_metrics=bundle['metrics'],
        training_summary=bundle['training_summary'],
    )


def predict_commission_for_booking(payload: dict[str, Any]) -> dict[str, Any]:
    bundle = load_model_bundle()
    history: pd.DataFrame = bundle['history']
    pipeline: Pipeline = bundle['pipeline']
    doctor_table, _, hospital_table = load_reference_tables()

    booking_date = pd.to_datetime(payload.get('booking_date'), errors='coerce')
    if pd.isna(booking_date):
        booking_date = history['Payment_date'].max() + pd.Timedelta(days=1)

    def fallback_numeric(column: str) -> float:
        return float(history[column].dropna().median())

    def fallback_text(column: str) -> str:
        return str(history[column].dropna().mode().iat[0])

    doctor_id = (payload.get('doctor_id') or '').strip()

    doctor_profile = None
    hospital_profile = None
    if doctor_id:
        doctor_matches = doctor_table[doctor_table['Doctor_id'] == doctor_id]
        if not doctor_matches.empty:
            doctor_profile = doctor_matches.iloc[0]
            hospital_matches = hospital_table[hospital_table['Hospital_id'] == doctor_profile['Hospital_id']]
            if not hospital_matches.empty:
                hospital_profile = hospital_matches.iloc[0]

    revenue_package = payload.get('revenue_package', '')
    
    # Priority 1: Direct consultation_fee and/or service_charge inputs
    if payload.get('consultation_fee') and payload.get('service_charge'):
        consultation_fees = float(payload['consultation_fee'])
        service_charge_val = float(payload['service_charge'])
    # Priority 2: Revenue value (will be split)
    elif payload.get('revenue'):
        total_revenue = float(payload['revenue'])
        avg_consultation_ratio = float(history['Consultation_fees'].sum() / (history['Consultation_fees'].sum() + history['Service_fee'].sum()))
        consultation_fees = total_revenue * avg_consultation_ratio
        service_charge_val = total_revenue * (1 - avg_consultation_ratio)
    # Priority 3: Individual consultation_fee or service_charge
    elif payload.get('consultation_fee') or payload.get('service_charge'):
        if payload.get('consultation_fee'):
            consultation_fees = float(payload['consultation_fee'])
        elif doctor_profile is not None and pd.notna(doctor_profile['Consultation_fees']):
            consultation_fees = float(doctor_profile['Consultation_fees'])
        else:
            consultation_fees = fallback_numeric('Consultation_fees')

        if payload.get('service_charge'):
            service_charge_val = float(payload['service_charge'])
        elif hospital_profile is not None and pd.notna(hospital_profile['Service_charge']):
            service_charge_val = float(hospital_profile['Service_charge'])
        else:
            service_charge_val = fallback_numeric('Service_charge')
    # Priority 4: Revenue package (legacy)
    elif revenue_package:
        consultation_fees, service_charge_val = revenue_package.split('|')
        consultation_fees = float(consultation_fees)
        service_charge_val = float(service_charge_val)
    # Priority 5: Doctor/hospital profiles
    elif doctor_profile is not None and pd.notna(doctor_profile['Consultation_fees']):
        consultation_fees = float(doctor_profile['Consultation_fees'])
        service_charge_val = (
            float(hospital_profile['Service_charge'])
            if hospital_profile is not None and pd.notna(hospital_profile['Service_charge'])
            else fallback_numeric('Service_charge')
        )
    # Fallback
    else:
        consultation_fees = fallback_numeric('Consultation_fees')
        service_charge_val = fallback_numeric('Service_charge')

    row = {
        'Consultation_fees': consultation_fees,
        'Age': float(payload.get('age') or fallback_numeric('Age')),
        'Gender': payload.get('gender') or fallback_text('Gender'),
        'payment_method': payload.get('payment_method') or fallback_text('payment_method'),
        'department': (
            str(doctor_profile['Specialization_group'])
            if doctor_profile is not None and pd.notna(doctor_profile['Specialization_group'])
            else payload.get('specialization') or fallback_text('department')
        ),
        'Experience_years': (
            float(doctor_profile['Experience_years'])
            if doctor_profile is not None and pd.notna(doctor_profile['Experience_years'])
            else float(payload.get('experience_years') or fallback_numeric('Experience_years'))
        ),
        'doctor_type': (
            'Online Doctor'
            if doctor_profile is not None and str(doctor_profile['Online_consultation']).strip() == 'Yes'
            else 'Offline Doctor'
            if doctor_profile is not None and str(doctor_profile['Online_consultation']).strip() == 'No'
            else payload.get('doctor_type') or fallback_text('doctor_type')
        ),
        'Rating_avg': (
            float(doctor_profile['Rating_avg'])
            if doctor_profile is not None and pd.notna(doctor_profile['Rating_avg'])
            else float(payload.get('rating_avg') or fallback_numeric('Rating_avg'))
        ),
        'Rating_count': (
            float(doctor_profile['Rating_count'])
            if doctor_profile is not None and pd.notna(doctor_profile['Rating_count'])
            else float(payload.get('rating_count') or fallback_numeric('Rating_count'))
        ),
        'city': (
            str(hospital_profile['District'])
            if hospital_profile is not None and pd.notna(hospital_profile['District'])
            else payload.get('city') or fallback_text('city')
        ),
        'Hospital_type': (
            str(hospital_profile['Hospital_type'])
            if hospital_profile is not None and pd.notna(hospital_profile['Hospital_type'])
            else payload.get('hospital_type') or fallback_text('Hospital_type')
        ),
        'Emergency_service': (
            str(hospital_profile['Emergency_service'])
            if hospital_profile is not None and pd.notna(hospital_profile['Emergency_service'])
            else payload.get('emergency_service') or fallback_text('Emergency_service')
        ),
        'Service_charge': service_charge_val,
        'booking_month': int(booking_date.month),
        'booking_day': int(booking_date.day),
        'booking_weekday': booking_date.day_name(),
        'booking_week_of_year': int(booking_date.isocalendar().week),
    }

    prediction_frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    predicted_commission_amount = round(float(pipeline.predict(prediction_frame)[0]), 2)
    predicted_commission_pct = round((predicted_commission_amount / consultation_fees) * 100, 2) if consultation_fees else 0.0

    return {
        'input': {
            'booking_date': booking_date.date().isoformat(),
            'doctor_id': doctor_id or 'All',
            'doctor_name': (
                str(doctor_profile['Doctor_name'])
                if doctor_profile is not None and pd.notna(doctor_profile['Doctor_name'])
                else 'All'
            ),
            'hospital_name': (
                str(hospital_profile['Hospital_name'])
                if hospital_profile is not None and pd.notna(hospital_profile['Hospital_name'])
                else 'All'
            ),
            **row,
        },
        'prediction': {
            'predicted_commission_amount': predicted_commission_amount,
            'predicted_commission_pct': predicted_commission_pct,
            'model_name': bundle['metrics']['model_name'],
        },
        'model_metrics': bundle['metrics'],
    }
