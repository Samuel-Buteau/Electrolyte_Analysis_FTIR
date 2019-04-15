from django.db import models


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

