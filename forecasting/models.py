from django.db import models


class ForecastRun(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    horizon_days = models.PositiveIntegerField(default=30)
    booking_growth_pct = models.FloatField(default=0.0)
    cancellation_rate = models.FloatField(default=0.0)
    projected_bookings = models.PositiveIntegerField(default=0)
    projected_commission = models.FloatField(default=0.0)
    baseline_commission = models.FloatField(default=0.0)
    uplift_commission = models.FloatField(default=0.0)
    metrics_snapshot = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self) -> str:
        return (
            f"ForecastRun({self.created_at:%Y-%m-%d %H:%M}, "
            f"commission={self.projected_commission:.2f})"
        )


class ForecastBreakdown(models.Model):
    forecast_run = models.ForeignKey(
        ForecastRun,
        related_name='breakdowns',
        on_delete=models.CASCADE,
    )
    dimension = models.CharField(max_length=50)
    segment = models.CharField(max_length=120)
    projected_commission = models.FloatField(default=0.0)
    projected_bookings = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ['dimension', '-projected_commission', 'segment']

    def __str__(self) -> str:
        return f"{self.dimension}:{self.segment}"
