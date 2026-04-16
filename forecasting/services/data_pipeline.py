from __future__ import annotations

import re

import pandas as pd
from django.conf import settings


def normalize_id(value: object, prefix: str, width: int = 3) -> object:
    if pd.isna(value):
        return pd.NA
    match = re.search(r'(\d+)', str(value))
    if not match:
        return pd.NA
    return f'{prefix}{match.group(1).zfill(width)}'


def load_reference_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    doctor = pd.read_csv(settings.FORECAST_DOCTOR_FILE)
    patient = pd.read_csv(settings.FORECAST_PATIENT_FILE)
    hospital = pd.read_csv(settings.FORECAST_HOSPITAL_FILE)

    doctor.columns = [col.strip() for col in doctor.columns]
    patient.columns = [col.strip() for col in patient.columns]
    hospital.columns = [col.strip() for col in hospital.columns]

    doctor = doctor[doctor['Doctor_id'].notna()].copy()
    doctor['Doctor_id'] = doctor['Doctor_id'].map(lambda value: normalize_id(value, 'DOC'))
    doctor['Hospital_id'] = doctor['Hospital_id'].map(lambda value: normalize_id(value, 'HOS'))
    patient['Doctor_id'] = patient['Doctor_id'].map(lambda value: normalize_id(value, 'DOC'))
    patient['Hospital_id'] = patient['Hospital_id'].map(lambda value: normalize_id(value, 'HOS'))
    hospital['Hospital_id'] = hospital['Hospital_id'].map(lambda value: normalize_id(value, 'HOS'))

    patient['Payment_date'] = pd.to_datetime(patient['Payment_date'], errors='coerce')

    for column in ['Consultation_fees', 'Service_fee', 'Age']:
        patient[column] = pd.to_numeric(patient[column], errors='coerce')
    for column in ['Age', 'Experience_years', 'Rating_avg', 'Rating_count', 'Total_reviews']:
        doctor[column] = pd.to_numeric(doctor[column], errors='coerce')
    for column in ['Service_charge', 'Total_beds', 'ICU_beds_available']:
        hospital[column] = pd.to_numeric(hospital[column], errors='coerce')

    return doctor, patient, hospital


def load_booking_dataset() -> pd.DataFrame:
    doctor, patient, hospital = load_reference_tables()

    merged = (
        patient.merge(
            doctor[
                [
                    'Doctor_id',
                    'Doctor_name',
                    'Specialization_group',
                    'Experience_years',
                    'Online_consultation',
                    'Rating_avg',
                    'Rating_count',
                    'Hospital_id',
                ]
            ],
            on='Doctor_id',
            how='left',
            suffixes=('', '_doctor'),
        ).merge(
            hospital[
                [
                    'Hospital_id',
                    'Hospital_name',
                    'District',
                    'Hospital_type',
                    'Emergency_service',
                    'Service_charge',
                ]
            ],
            on='Hospital_id',
            how='left',
        )
    )

    merged = merged.sort_values('Payment_date').reset_index(drop=True)
    merged['commission_amount'] = merged['Service_fee'].fillna(0.0)
    merged['commission_pct'] = (
        merged['Service_fee'].div(merged['Consultation_fees']).replace([pd.NA, pd.NaT], 0.0) * 100
    ).fillna(0.0)
    merged['department'] = merged['Specialization_group'].fillna('Unknown')
    merged['city'] = merged['District'].fillna('Unknown')
    merged['doctor_type'] = merged['Online_consultation'].fillna('Unknown').map(
        {'Yes': 'Online Doctor', 'No': 'Offline Doctor'}
    ).fillna('Unknown')
    merged['payment_method'] = merged['Payment_type'].fillna('Unknown')
    merged['doctor_name'] = merged['Doctor_name'].fillna('Unknown')
    merged['hospital_name'] = merged['Hospital_name'].fillna('Unknown')
    merged['booking_month'] = merged['Payment_date'].dt.month.fillna(0).astype(int)
    merged['booking_day'] = merged['Payment_date'].dt.day.fillna(0).astype(int)
    merged['booking_weekday'] = merged['Payment_date'].dt.day_name().fillna('Unknown')
    merged['booking_week_of_year'] = (
        merged['Payment_date'].dt.isocalendar().week.astype('Int64').fillna(0).astype(int)
    )

    return merged
