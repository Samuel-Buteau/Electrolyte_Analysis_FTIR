from django.core.management.base import BaseCommand
import numpy
wanted_wavenumbers = []
from FTIR_to_electrolyte_composition.models import FTIRSpectrum, FTIRSample, HUMAN,ROBOT,CELL
import matplotlib.pyplot as plt

import os


with open(os.path.join('.','Data','WantedWavenumbers.csv') ,'r') as f:
    wanted_wavenumbers = [float(x) for x in f.read().split('\n')[:-1]]

number_inputs = 1536
def ImportDirect(file):
    tags = ['3596','3999','649','1','2','4']
    n_total = 3596
    pre_counter = 0
    raw_data = n_total * [0.0]
    counter = 0
    for _ in range(5000):

        my_line = file.readline()
        if pre_counter<len(tags):
            if not my_line.startswith(tags[pre_counter]):
                print("unrecognized format", my_line)
            pre_counter += 1
            continue

        if counter >= n_total or my_line == '':
            break

        raw_data[counter] = float(my_line.split('\n')[0])
        counter +=1

    just_important_data = number_inputs * [0.0]
    for i in range(number_inputs):
        just_important_data[i] = raw_data[-1 - i]


    return numpy.array(just_important_data)



elizabeth_meta = {
"1_EYL_DEC_PURE.asp":(('m', 0), {'DEC':1}),
"2_EYL_DEC_0.5m.asp":(('m', 0.5), {'DEC':1}),
"3_EYL_DEC_1m.asp":(('m', 1), {'DEC':1}),
"4_EYL_DEC_1.5m.asp":(('m', 1.5), {'DEC':1}),

"5_EYL_EC-DEC_1-2_PURE.asp":(('m', 0), {'EC':1,'DEC':2}),
"6_EYL_EC-DEC_1-2_0.5m.asp":(('m', 0.5), {'EC':1,'DEC':2}),
"7_EYL_EC-DEC_1-2_1m.asp":(('m', 1), {'EC':1,'DEC':2}),
"8_EYL_EC-DEC_1-2_1.5m.asp":(('m', 1.5), {'EC':1,'DEC':2}),

"9_EYL_EC-EMC-DMC_10-10-80_PURE.asp":(('m', 0), {'EC':10,'EMC':10,'DMC':80}),
"10_EYL_EC-EMC-DMC_10-10-80_0.5m.asp":(('m', 0.5), {'EC':10,'EMC':10,'DMC':80}),
"11_EYL_EC-EMC-DMC_10-10-80_1m.asp":(('m', 1.0), {'EC':10,'EMC':10,'DMC':80}),
"12_EYL_EC-EMC-DMC_10-10-80_1.5m.asp":(('m', 1.5), {'EC':10,'EMC':10,'DMC':80}),
"13_EYL_EMC_PURE.asp":(('m', 0), {'EMC':1}),
"14_EYL_EMC_0.5m.asp":(('m', 0.5), {'EMC':1}),
"15_EYL_EMC_1m.asp":(('m', 1), {'EMC':1}),
"16_EYL_EMC_1.5m.asp":(('m', 1.5), {'EMC':1}),

"17_EYL_EC-DEC_1-1_PURE.asp":(('m', 0), {'EC':1,'DEC':1}),
"18_EYL_EC-DEC_1-1_0.5m.asp":(('m', 0.5), {'EC':1,'DEC':1}),
"19_EYL_EC-DEC_1-1_1m.asp":(('m', 1), {'EC':1,'DEC':1}),
"20_EYL_EC-DEC_1-1_1.5m.asp":(('m', 1.5), {'EC':1,'DEC':1}),
"21_EYL_EC-DMC_1-1_PURE.asp":(('m', 0), {'EC':1,'DMC':1}),
"22_EYL_EC-DMC_1-1_0.5m.asp":(('m', 0.5), {'EC':1,'DMC':1}),
"23_EYL_EC-DMC_1-1_1m.asp":(('m', 1), {'EC':1,'DMC':1}),
"24_EYL_EC-DMC_1-1_1.5m.asp":(('m', 1.5), {'EC':1,'DMC':1}),

"25_EYL_EC-EMC_3-7_PURE.asp":(('m', 0), {'EC':3,'EMC':7}),
"26_EYL_EC-EMC_3-7_0.5m.asp":(('m', 0.5), {'EC':3,'EMC':7}),
"27_EYL_EC-EMC_3-7_1m.asp":(('m', 1), {'EC':3,'EMC':7}),
"28_EYL_EC-EMC_3-7_1.5m.asp":(('m', 1.5), {'EC':3,'EMC':7}),

"29_EYL_EC-EMC-DMC_25-5-70_PURE_FROM_NEW_BOTTLE.asp":(('m', 0), {'EC':25,'EMC':5,'DMC':70}),
"30_EYL_EC-EMC-DMC_25-5-70_0.5m.asp":(('m', 0.5), {'EC':25,'EMC':5,'DMC':70}),
"31_EYL_EC-EMC-DMC_25-5-70_1m.asp":(('m', 1), {'EC':25,'EMC':5,'DMC':70}),
"32_EYL_EC-EMC-DMC_25-5-70_1.5m.asp":(('m', 1.5), {'EC':25,'EMC':5,'DMC':70}),

"33_EYL_EC-EMC-DMC_25-5-70_PURE_FROM_OLD_BOTTLE.asp":(('m', 0), {'EC':25,'EMC':5,'DMC':70}),
"34_EYL_EC-EMC-DMC_25-5-70_0.5M.asp":(('M', 1.1,0.08), {'EC':25,'EMC':5,'DMC':70}),
"35_EYL_EC-EMC-DMC_25-5-70_1M.asp":(('M', 1.1,0.15), {'EC':25,'EMC':5,'DMC':70}),
"36_EYL_EC-EMC-DMC_25-5-70_1.5M.asp":(('M', 1.1,0.23), {'EC':25,'EMC':5,'DMC':70}),

"37_EYL_DMC_PURE.asp":(('m', 0), {'DMC':1}),
"38_EYL_DMC_0.5M.asp":(('M', 1.07,0.08), {'DMC':1}),
"39_EYL_DMC_1M.asp":(('M', 1.07,0.15), {'DMC':1}),
"40_EYL_DMC_1.5M.asp":(('M', 1.07,0.23), {'DMC':1}),

"41_EYL_EC-DMC_3-7_PURE.asp":(('m', 0), {'EC':3,'DMC':7}),
"42_EYL_EC-DMC_3-7_0.5M.asp":(('M', 1.145,0.08), {'EC':3,'DMC':7}),
"43_EYL_EC-DMC_3-7_1M.asp":(('M', 1.145,0.15), {'EC':3,'DMC':7}),
"44_EYL_EC-DMC_3-7_1.5M.asp":(('M', 1.145,0.23), {'EC':3,'DMC':7}),

"45_EYL_EC-EMC_3-7_PURE.asp":(('m', 0), {'EC':3,'EMC':7}),
"46_EYL_EC-EMC_3-7_0.5M.asp":(('M', 1.09,0.08), {'EC':3,'EMC':7}),
"47_EYL_EC-EMC_3-7_1M.asp":(('M', 1.09,0.15), {'EC':3,'EMC':7}),
"48_EYL_EC-EMC_3-7_1.5M.asp":(('M', 1.09,0.23), {'EC':3,'EMC':7}),

"33b_EYL_EC-EMC-DMC_25-5-70_PURE_FROM_OLD_BOTTLE.asp":(('m', 0), {'EC':25,'EMC':5,'DMC':70}),
"34b_EYL_EC-EMC-DMC_25-5-70_0.5M.asp":(('M', 1.1,0.08), {'EC':25,'EMC':5,'DMC':70}),
"35b_EYL_EC-EMC-DMC_25-5-70_1M.asp":(('M', 1.1,0.15), {'EC':25,'EMC':5,'DMC':70}),
"36b_EYL_EC-EMC-DMC_25-5-70_1.5M.asp":(('M', 1.1,0.23), {'EC':25,'EMC':5,'DMC':70}),

"37b_EYL_DMC_PURE.asp":(('m', 0), {'DMC':1}),
"38b_EYL_DMC_0.5M.asp":(('M', 1.07,0.08), {'DMC':1}),
"39b_EYL_DMC_1M.asp":(('M', 1.07,0.15), {'DMC':1}),
"40b_EYL_DMC_1.5M.asp":(('M', 1.07,0.23), {'DMC':1}),

"41b_EYL_EC-DMC_3-7_PURE.asp":(('m', 0), {'EC':3,'DMC':7}),
"42b_EYL_EC-DMC_3-7_0.5M.asp":(('M', 1.145,0.08), {'EC':3,'DMC':7}),
"43b_EYL_EC-DMC_3-7_1M.asp":(('M', 1.145,0.15), {'EC':3,'DMC':7}),
"44b_EYL_EC-DMC_3-7_1.5M.asp":(('M', 1.145,0.23), {'EC':3,'DMC':7}),

"45b_EYL_EC-EMC_3-7_PURE.asp":(('m', 0), {'EC':3,'EMC':7}),
"46b_EYL_EC-EMC_3-7_0.5M.asp":(('M', 1.09,0.08), {'EC':3,'EMC':7}),
"47b_EYL_EC-EMC_3-7_1M.asp":(('M', 1.09,0.15), {'EC':3,'EMC':7}),
"48b_EYL_EC-EMC_3-7_1.5M.asp":(('M', 1.09,0.23), {'EC':3,'EMC':7}),


}

mass_per_mol_LiPF6 = 0.151905
def from_metadata_to_mass_ratios(meta):
    if meta[0][0] == 'm':
        # mode molal
        mass_of_salt_per_total = meta[0][1] * mass_per_mol_LiPF6
    elif meta[0][0] == 'M':
        #mode Molar
        mass_of_salt_per_total = meta[0][2]/meta[0][1]

    mass_of_solvents_per_total = 1.-mass_of_salt_per_total

    if not 'EC' in meta[1].keys():
        mass_of_ec_per_total = 0.0
    else:
        mass_of_ec_per_total = meta[1]['EC']

    if not 'EMC' in meta[1].keys():
        mass_of_emc_per_total = 0.0
    else:
        mass_of_emc_per_total = meta[1]['EMC']


    if not 'DMC' in meta[1].keys():
        mass_of_dmc_per_total = 0.0
    else:
        mass_of_dmc_per_total = meta[1]['DMC']

    if not 'DEC' in meta[1].keys():
        mass_of_dec_per_total = 0.0
    else:
        mass_of_dec_per_total = meta[1]['DEC']


    mass_of_solvents = (mass_of_ec_per_total +
                        mass_of_emc_per_total +
                        mass_of_dmc_per_total +
                        mass_of_dec_per_total
                        )

    mass_of_ec_per_total *=(mass_of_solvents_per_total/
                            mass_of_solvents)

    mass_of_emc_per_total *=(mass_of_solvents_per_total/
                            mass_of_solvents)

    mass_of_dmc_per_total *= (mass_of_solvents_per_total /
                             mass_of_solvents)

    mass_of_dec_per_total *= (mass_of_solvents_per_total /
                             mass_of_solvents)


    return {'LiPF6':mass_of_salt_per_total,
            'EC':mass_of_ec_per_total,
            'EMC': mass_of_emc_per_total,
            'DMC': mass_of_dmc_per_total,
            'DEC': mass_of_dec_per_total
            }

update_eli = True
update_robot = True
class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        if update_eli:


            for filename in  elizabeth_meta.keys():
                meta = elizabeth_meta[filename]
                if not os.path.exists(os.path.join('.','Data','Elizabeth', filename)):
                    continue
                if FTIRSpectrum.objects.filter(preparation=HUMAN,filename=filename):
                    continue

                with open(os.path.join('.','Data','Elizabeth', filename), 'r') as f:
                    dat = ImportDirect(f)

                ratios = from_metadata_to_mass_ratios(meta)

                spec = FTIRSpectrum(filename=filename,
                                    preparation=HUMAN,
                                    LIPF6_mass_ratio = ratios['LiPF6'],
                                    EC_mass_ratio=ratios['EC'],
                                    EMC_mass_ratio=ratios['EMC'],
                                    DMC_mass_ratio=ratios['DMC'],
                                    DEC_mass_ratio=ratios['DEC'],
                                    supervised = True)

                spec.save()

                print('ratio:,',ratios)
                print('data: ', dat)

                samps = []
                for index in range(len(dat)):
                    samps.append(FTIRSample(
                        spectrum=spec,
                        index=index,
                        wavenumber=wanted_wavenumbers[index],
                        absorbance=dat[index]
                    ))

                FTIRSample.objects.bulk_create(samps)

        if update_robot:


            all_filenames = []
            path_to_robot = os.path.join('.','Data', 'Robot')
            for root, dirs, filenames in os.walk(path_to_robot):
                for file in filenames:
                    if file.endswith('.asp'):
                        all_filenames.append(os.path.join(root, file))

            for filename in all_filenames:
                if FTIRSpectrum.objects.filter(preparation=ROBOT, filename=filename):
                    continue

                with open(filename, 'r') as f:
                    dat = ImportDirect(f)

                spec = FTIRSpectrum(filename=filename,
                                    preparation=ROBOT,
                                    supervised=False)

                spec.save()

                print('data: ', dat)
                # plt.plot(range(len(dat)), dat)
                # plt.show()
                samps = []
                for index in range(len(dat)):
                    samps.append(FTIRSample(
                        spectrum=spec,
                        index=index,
                        wavenumber=wanted_wavenumbers[index],
                        absorbance=dat[index]
                    ))

                FTIRSample.objects.bulk_create(samps)




