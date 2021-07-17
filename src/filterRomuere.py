from skimage import segmentation, color
from skimage.future import graph
from skimage.io import imread,imsave

class filterRomuere():
    """
    Aplica um filter de redução de cores em uma imagem.

    Parametros
    ----------
    imagem : str
        Endereço da imagem de entrada.
        A imagem de entrada deve não ter valores NaN
    """

    def __init__(self,imagem) -> None:
        self._path = imagem
        self._imagem = imread(imagem)

    def apply_filter(self,compactness=30, n_segments=400, status_label=1):
        """
        compactness: float, optional
            Proximidade do espaço de cor.
        n_segments : int, optional
            O número aproximado de rótulos na imagem de saida.
        start_label : int, optional
            Índice do rótulo inicial.

        Returns
        -------
        out2 : array of float, shape (M, N, 3)
            Image colorida.
        """

        label = segmentation.slic(self._imagem, compactness=30, n_segments=400, start_label=1)
        g = graph.rag_mean_color(self._imagem, labels1, mode='similarity')
        labels2 = graph.cut_normalized(labels2, self._imagem, kind='avg', bg_label=0)
        imsave(self._path[:-4]+'_saida.png',out2)