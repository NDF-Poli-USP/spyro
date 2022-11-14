import numpy as np
from . import eikCrit_spy
from . import damping_spy

# Work from Ruben Andres Salas,
# Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva, Hybrid absorbing scheme based on hyperel-
# liptical layers with non-reflecting boundary conditions in scalar wave equations, Applied Mathematical
# Modelling (2022), doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender

class HABC:
    """ class HABC that determines absorbing layer size and parameters to be used
    """
    def __init__(self, Wave_object, histPcrit=None, f0, it_fwi):
        """Initializes class and gets a wave object as an input.

        Parameters
        ----------
        Wave_object: `dictionary`
            Contains simulation parameters and options.

        Returns
        -------
        pad_length; size of absorving layer

        """

        ''''
        from fenics import Point


        def receiver(t, posrec, w):
            """
            Data aquisition of the receivers

            *Input arguments
                t: Current time
                posrec: Receivers localization
                w: Variable of state
            *Output arguments
                datarec: Receiver history no current time t
            """
            print("Recording at Receivers")
            datarec = []
            datarec += [t]

            for i in range(posrec.shape[1]):
                datarec += [w(Point(posrec[0, i], posrec[1, i]))]

            return datarec


        # Output files
        histrec += rc.receiver(t, Dom.posrec, w)
        histrec = np.reshape(histrec,
                        (len(histrec)//(Dom.posrec.shape[1]+1),
                        Dom.posrec.shape[1]+1))
        '''
        Lz = Wave_object.length_z
        Lx = Wave_object.length_x
        zs =[]
        xs = []
        for source in Wave_object.model_parameters.source_locations:
            z,x = source
            zs.append(z)
            xs.append(x)

        possou = [zs, xs]
        Z, posCrit, cref = eikCrit_spy.Eikonal(Wave_object)

        self.Lz = Lz
        self.Lx = Lx
        self.possou = possou
        self.posCrit = posCrit
        self.histPcrit = histPcrit


        fref, F_L, pad_length = habc_size(Lz, Lx, posCrit, possou, f0, it_fwi, lmin, Z,  histPcrit=None,TipLay='REC', nexp=np.nan)