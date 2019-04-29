from django.db import models

'''
Below are defined the formats in which the data is stored into the database. 

The database is as follows:
- there are potentially a few experiments
- each experiment can contain many spectra.
- each spectrum contains many samples.



An Experiment is defined with a name.

An FTIRSpectrum is defined as follows:
- a filename,
- an experiment (this is optional).
- a preparation method, either Human, Robot, or Cell.
- the mass ratios for LiPF6, EC, EMC, DMC, DEC (this is optional)
- supervised is true if the mass ratios should be used as known values for training and cross-validation. If false, it means that these values are either not known or not exactly trusted.


an FTIRSample corresponds to a single wavenumber, and a spectrum is made of many such FTIRSamples. it is defined as follows:
- an FTIRSpectrum
- an index going from 1 to 1536 (this increases monotonically with wavenumber).
- the wavenumber for the sample in inverse centimeters
- the absorbance measured for that wavenumber and spectrum. 



Here are a few examples of ways to add data to the database:

to check if a spectrum with filename filename and preparation HUMAN already exists:
if FTIRSpectrum.objects.filter(preparation=HUMAN,filename=filename):
    # do something if exists
    
to record a spectrum with known mass ratios:

#first record the FTIRSpectrum
spec = FTIRSpectrum(filename=filename,
                    preparation=HUMAN,
                    LIPF6_mass_ratio = ratios['LiPF6'],
                    EC_mass_ratio=ratios['EC'],
                    EMC_mass_ratio=ratios['EMC'],
                    DMC_mass_ratio=ratios['DMC'],
                    DEC_mass_ratio=ratios['DEC'],
                    supervised = True)

# save it to database
spec.save()

#then record a list of samples.
samps = []
for index in range(len(dat)):
    samps.append(FTIRSample(
        spectrum=spec,
        index=index,
        wavenumber=wanted_wavenumbers[index],
        absorbance=dat[index]
    ))

# and add them all at once to the database.
FTIRSample.objects.bulk_create(samps)



to record a spectrum with unknown mass ratios:

spec = FTIRSpectrum(filename=filename,
                    preparation=ROBOT,
                    supervised=False)

spec.save()

samps = []
for index in range(len(dat)):
    samps.append(FTIRSample(
        spectrum=spec,
        index=index,
        wavenumber=wanted_wavenumbers[index],
        absorbance=dat[index]
    ))

FTIRSample.objects.bulk_create(samps)





then, in order to access the spectra that were supervised (e.g. for training)

# make a python dictionary with 's' for samples, 'm' for mass ratios, and 'z' for supervised flag.
supervised_dataset = {'s': [], 'm': [], 'z': []}



#simply iterate through all spectra which are supervised
for spec in FTIRSpectrum.objects.filter(supervised=True):
    #set flag
    supervised_dataset['z'].append(1.)
    #access and record all mass ratios
    supervised_dataset['m'].append([spec.LIPF6_mass_ratio, spec.EC_mass_ratio, spec.EMC_mass_ratio,
                                    spec.DMC_mass_ratio, spec.DEC_mass_ratio])
    
    
    #make a vector of the absorbances (simply take the samples with the right spectrum.)    
    supervised_dataset['s'].append(
        [samp.absorbance for samp in FTIRSample.objects.filter(spectrum=spec).order_by('index')])

# make numpy arrays for speed and interplay with tensorflow.
supervised_dataset['s'] = numpy.array(supervised_dataset['s'])
supervised_dataset['m'] = numpy.array(supervised_dataset['m'])
supervised_dataset['z'] = numpy.array(supervised_dataset['z'])

'''



HUMAN = 'H'
ROBOT = 'R'
CELL = 'C'
preparation_Choices = (
    (HUMAN, 'Human'),
    (ROBOT, 'Robot'),
    (CELL, 'Cell')
)

class Experiment(models.Model):
    name = models.CharField(max_length=1000)


class FTIRSpectrum(models.Model):

    filename = models.CharField(max_length=1000)
    experiment = models.ForeignKey(Experiment, on_delete=models.SET_NULL, null=True)
    preparation = models.CharField(max_length=1, choices = preparation_Choices)
    LIPF6_mass_ratio = models.FloatField(blank=True, null=True)
    EC_mass_ratio = models.FloatField(blank=True, null=True)
    EMC_mass_ratio = models.FloatField(blank=True, null=True)
    DMC_mass_ratio = models.FloatField(blank=True, null=True)
    DEC_mass_ratio = models.FloatField(blank=True, null=True)
    supervised = models.BooleanField(default=False)


class FTIRSample(models.Model):
    spectrum = models.ForeignKey(FTIRSpectrum, on_delete=models.CASCADE)
    index = models.IntegerField()
    wavenumber = models.FloatField()
    absorbance = models.FloatField()

