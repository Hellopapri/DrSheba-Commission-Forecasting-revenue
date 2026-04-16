from django.core.management.base import BaseCommand

from forecasting.services.modeling import train_and_persist_model_bundle


class Command(BaseCommand):
    help = 'Train the commission forecasting model and persist artifacts.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Retrain even if an artifact already exists.',
        )

    def handle(self, *args, **options):
        bundle = train_and_persist_model_bundle(force_retrain=options['force'])
        metrics = bundle['metrics']
        training_summary = bundle['training_summary']
        self.stdout.write(
            self.style.SUCCESS(
                'Trained model '
                f"{metrics['model_name']} | "
                f"MAE={metrics['mae']} | RMSE={metrics['rmse']} | "
                f"MAPE={metrics['mape']}% | R2={metrics['r2']} | "
                f"Bookings={training_summary['bookings']}"
            )
        )
