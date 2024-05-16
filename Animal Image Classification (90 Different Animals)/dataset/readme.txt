dataset link - https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals

<------------- To download the Datset----------->


!pip install opendatasets --q

import opendatasets as od
dataset_url = "https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals"
od.download(dataset_url)


