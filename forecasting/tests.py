from django.test import Client, TestCase

from forecasting.services.modeling import train_and_persist_model_bundle


class ForecastApiTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        train_and_persist_model_bundle(force_retrain=True)

    def setUp(self):
        self.client = Client()

    def test_summary_endpoint_returns_metrics_and_breakdowns(self):
        response = self.client.get('/api/forecast/summary/', {'horizon_days': 15})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn('summary', payload)
        self.assertIn('model_metrics', payload)
        self.assertIn('breakdowns', payload)
        self.assertGreater(payload['summary']['projected_bookings'], 0)

    def test_scenario_endpoint_persists_forecast_run(self):
        response = self.client.post(
            '/api/forecast/scenario/',
            data='{"horizon_days": 30, "booking_growth_pct": 10, "cancellation_rate": 5}',
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload['saved'])

    def test_booking_prediction_endpoint_returns_single_prediction(self):
        response = self.client.get(
            '/api/forecast/booking-predict/',
            {
                'booking_date': '2026-05-01',
                'consultation_fees': 1500,
                'doctor_id': 'DOC001',
                'payment_method': 'Bkash',
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn('prediction', payload)
        self.assertGreater(payload['prediction']['predicted_commission_amount'], 0)
        self.assertEqual(payload['input']['doctor_id'], 'DOC001')
