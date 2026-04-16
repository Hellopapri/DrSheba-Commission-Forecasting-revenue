from django.contrib import admin

from .models import ForecastBreakdown, ForecastRun


class ForecastBreakdownInline(admin.TabularInline):
    model = ForecastBreakdown
    extra = 0
    readonly_fields = ('dimension', 'segment', 'projected_commission', 'projected_bookings')


@admin.register(ForecastRun)
class ForecastRunAdmin(admin.ModelAdmin):
    list_display = (
        'created_at',
        'horizon_days',
        'booking_growth_pct',
        'cancellation_rate',
        'projected_commission',
        'uplift_commission',
    )
    list_filter = ('horizon_days',)
    readonly_fields = (
        'created_at',
        'projected_bookings',
        'projected_commission',
        'baseline_commission',
        'uplift_commission',
        'metrics_snapshot',
    )
    inlines = [ForecastBreakdownInline]
