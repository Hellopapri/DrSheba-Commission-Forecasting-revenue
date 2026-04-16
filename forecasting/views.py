from __future__ import annotations

import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from .models import ForecastBreakdown, ForecastRun
from .services.modeling import (
    generate_forecast,
    get_forecast_options,
    predict_commission_for_booking,
    train_and_persist_model_bundle,
)


def _parse_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _parse_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _build_response_payload(horizon_days: int, booking_growth_pct: float, cancellation_rate: float) -> dict:
    return _build_response_payload_with_options(
        horizon_days=horizon_days,
        booking_growth_pct=booking_growth_pct,
        cancellation_rate=cancellation_rate,
        avg_daily_bookings=None,
        filters={},
    )


def _build_response_payload_with_options(
    horizon_days: int,
    booking_growth_pct: float,
    cancellation_rate: float,
    avg_daily_bookings: float | None,
    filters: dict[str, str],
) -> dict:
    result = generate_forecast(
        horizon_days=horizon_days,
        booking_growth_pct=booking_growth_pct,
        cancellation_rate=cancellation_rate,
        avg_daily_bookings=avg_daily_bookings,
        filters=filters,
    )
    return {
        'summary': result.summary,
        'breakdowns': result.breakdowns,
        'model_metrics': result.model_metrics,
        'training_summary': result.training_summary,
    }


@require_GET
def dashboard_view(request):
    train_and_persist_model_bundle(force_retrain=False)

    forecast_form = {
        'horizon_days': _parse_int(request.GET.get('forecast_horizon_days'), 30),
        'booking_growth_pct': _parse_float(request.GET.get('forecast_booking_growth_pct'), 10.0),
        'cancellation_rate': _parse_float(request.GET.get('forecast_cancellation_rate'), 5.0),
        'avg_daily_bookings': _parse_float(request.GET.get('forecast_avg_daily_bookings'), 1.0),
        'department': request.GET.get('forecast_department', ''),
        'city': request.GET.get('forecast_city', ''),
        'doctor_type': request.GET.get('forecast_doctor_type', ''),
        'payment_method': request.GET.get('forecast_payment_method', ''),
    }
    filters = {
        'department': forecast_form['department'],
        'city': forecast_form['city'],
        'doctor_type': forecast_form['doctor_type'],
        'payment_method': forecast_form['payment_method'],
    }
    payload = _build_response_payload_with_options(
        horizon_days=forecast_form['horizon_days'],
        booking_growth_pct=forecast_form['booking_growth_pct'],
        cancellation_rate=forecast_form['cancellation_rate'],
        avg_daily_bookings=forecast_form['avg_daily_bookings'],
        filters=filters,
    )

    booking_form = {
        'booking_date': request.GET.get('booking_date', ''),
        'doctor_id': request.GET.get('doctor_id', ''),
        'payment_method': request.GET.get('payment_method', ''),
        'specialization': request.GET.get('specialization', ''),
        'city': request.GET.get('city', ''),
        'revenue': request.GET.get('revenue', ''),
        'consultation_fee': request.GET.get('consultation_fee', ''),
        'service_charge': request.GET.get('service_charge', ''),
    }
    booking_prediction = None
    if request.GET.get('predict_booking') == '1':
        booking_prediction = predict_commission_for_booking(booking_form)

    context = {
        **payload,
        'filter_options': get_forecast_options(),
        'forecast_form': forecast_form,
        'booking_form': booking_form,
        'booking_prediction': booking_prediction,
    }
    return render(request, 'forecasting/dashboard.html', context)


@require_GET
def forecast_summary_view(request):
    horizon_days = _parse_int(request.GET.get('horizon_days'), 30)
    booking_growth_pct = _parse_float(request.GET.get('booking_growth_pct'), 0.0)
    cancellation_rate = _parse_float(request.GET.get('cancellation_rate'), 0.0)
    avg_daily_bookings = _parse_float(request.GET.get('avg_daily_bookings'), 0.0)
    filters = {
        'department': request.GET.get('department', ''),
        'city': request.GET.get('city', ''),
        'doctor_type': request.GET.get('doctor_type', ''),
        'payment_method': request.GET.get('payment_method', ''),
    }
    payload = _build_response_payload_with_options(
        horizon_days,
        booking_growth_pct,
        cancellation_rate,
        avg_daily_bookings if avg_daily_bookings > 0 else None,
        filters,
    )
    return JsonResponse(payload)


@csrf_exempt
@require_http_methods(['GET', 'POST'])
def forecast_scenario_view(request):
    if request.method == 'POST':
        body = json.loads(request.body or '{}')
        horizon_days = int(body.get('horizon_days', 30))
        booking_growth_pct = float(body.get('booking_growth_pct', 0.0))
        cancellation_rate = float(body.get('cancellation_rate', 0.0))
        avg_daily_bookings = float(body.get('avg_daily_bookings', 0.0))
        filters = {
            'department': body.get('department', ''),
            'city': body.get('city', ''),
            'doctor_type': body.get('doctor_type', ''),
            'payment_method': body.get('payment_method', ''),
        }
    else:
        horizon_days = _parse_int(request.GET.get('horizon_days'), 30)
        booking_growth_pct = _parse_float(request.GET.get('booking_growth_pct'), 10.0)
        cancellation_rate = _parse_float(request.GET.get('cancellation_rate'), 5.0)
        avg_daily_bookings = _parse_float(request.GET.get('avg_daily_bookings'), 0.0)
        filters = {
            'department': request.GET.get('department', ''),
            'city': request.GET.get('city', ''),
            'doctor_type': request.GET.get('doctor_type', ''),
            'payment_method': request.GET.get('payment_method', ''),
        }

    payload = _build_response_payload_with_options(
        horizon_days,
        booking_growth_pct,
        cancellation_rate,
        avg_daily_bookings if avg_daily_bookings > 0 else None,
        filters,
    )
    run = ForecastRun.objects.create(
        horizon_days=payload['summary']['horizon_days'],
        booking_growth_pct=payload['summary']['booking_growth_pct'],
        cancellation_rate=payload['summary']['cancellation_rate'],
        projected_bookings=payload['summary']['projected_bookings'],
        projected_commission=payload['summary']['projected_commission'],
        baseline_commission=payload['summary']['baseline_commission'],
        uplift_commission=payload['summary']['uplift_commission'],
        metrics_snapshot=payload['model_metrics'],
    )

    for dimension, records in payload['breakdowns'].items():
        for record in records:
            segment_key = next(key for key in record.keys() if key not in {'projected_commission', 'projected_bookings'})
            ForecastBreakdown.objects.create(
                forecast_run=run,
                dimension=dimension,
                segment=str(record[segment_key]),
                projected_commission=record['projected_commission'],
                projected_bookings=record['projected_bookings'],
            )

    payload['saved'] = True
    payload['forecast_run_id'] = run.pk
    return JsonResponse(payload)


@csrf_exempt
@require_http_methods(['GET', 'POST'])
def booking_prediction_view(request):
    if request.method == 'POST':
        payload = json.loads(request.body or '{}')
    else:
        payload = request.GET.dict()
    prediction = predict_commission_for_booking(payload)
    return JsonResponse(prediction)
