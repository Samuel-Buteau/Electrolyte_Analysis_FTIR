from django.core.management.base import BaseCommand
import numpy
wanted_wavenumbers = []
from FTIR_to_electrolyte_composition.models import FTIRSpectrum, FTIRSample, HUMAN,ROBOT,CELL
import matplotlib.pyplot as plt





update_eli = True
update_robot = False
class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        if update_eli:
            for spec in FTIRSpectrum.objects.filter(preparation=HUMAN):
                print('filename: ', spec.filename, 'Supervized? ', spec.supervised)
                print('LiPF6: {}, EC: {}, EMC: {}, DMC: {}, DEC: {}'.format(
                    spec.LIPF6_mass_ratio,
                    spec.EC_mass_ratio,
                    spec.EMC_mass_ratio,
                    spec.DMC_mass_ratio,
                    spec.DEC_mass_ratio))

                samples = FTIRSample.objects.filter(spectrum=spec).order_by('index')
                plt.scatter([s.wavenumber for s in samples], [s.absorbance for s in samples])
                plt.show()
                x = input('Please enter KEEP/DELETE/UNSUPERVISED:')
                if 'KEEP' in x:
                    continue
                elif 'DELETE' in x:
                    spec.delete()
                elif 'UNSUPERVISED' in x:
                    spec.supervised = False
                    spec.save()
                else:
                    continue

