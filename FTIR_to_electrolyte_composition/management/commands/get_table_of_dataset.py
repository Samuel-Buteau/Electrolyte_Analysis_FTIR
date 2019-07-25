from django.core.management.base import BaseCommand
import numpy
import csv
wanted_wavenumbers = []
from FTIR_to_electrolyte_composition.models import FTIRSpectrum, FTIRSample, HUMAN,ROBOT,CELL

class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        with open('table_of_dataset.csv', 'w',newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow([
                'LiPF6 mass ratio',
                'EC mass ratio',
                'EMC mass ratio',
                'DMC mass ratio',
                'DEC mass ratio',
            ])

            for spec in FTIRSpectrum.objects.filter(preparation=HUMAN,supervised=True):
                spamwriter.writerow([
                    '{:1.4f}'.format(spec.LIPF6_mass_ratio),
                    '{:1.4f}'.format(spec.EC_mass_ratio),
                    '{:1.4f}'.format(spec.EMC_mass_ratio),
                    '{:1.4f}'.format(spec.DMC_mass_ratio),
                    '{:1.4f}'.format(spec.DEC_mass_ratio)
                ])



