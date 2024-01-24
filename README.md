# Constrained K-Means
Il progetto si tratta di un'analisi sperimentale di un approccio innovativo per un problema di clustering vincolato. In particolare i vincoli determinano un problema di Hard Clustering bilanciato.
## K-Means
K-Means è uno degli algoritmi di clustering più conosciuti. Si tratta di un metodo molto semplice che permette di raggiungere in tempi brevi una soluzione sub-ottima accettabile. Ha lo svantaggio di non essere un algoritmo ottimo e di dipendere fortemente dalle condizioni iniziali.
## MIQP (Mixed Integer Quadratic Programming)
Questa metodologia permette di formulare il problema di clustering vincolato in un modo totalmente diverso. L'algoritmo che si costruisce è capace di raggiungere l'ottimo globale del problema con ogni configuarzione, ovviamente a discapito della velocità di esecuzione. Informazioni più approfondite a riguardo sono descritte nella directory Documentation di questa repository.
## Esperimenti
Per replicare i risultati, una volta clonata la repository basta selezionare in main.py il dataset che si vuole utilizzare tra quelli disponibili, dopodiché, ancora in main.py, selezionare il test che si vuole effettuare 
* (DATASET = 1: Synthetic 1, 2: Synthetic 2, 3: Heart Disease, 4: Coverage Type)
* (TEST = 1: al variare del numero di elementi, 2: al variare del numero di features, 3: al variare del numero di cluster).
