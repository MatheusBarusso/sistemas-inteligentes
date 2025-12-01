## O código de exemplo irá carregar a MobileNet e aplicá-la sobre as imagens. O resultado está correto? Por que?

 O exemplo ``` beardetector.py ``` utiliza a rede neural convolucional  **MobileNetV2**, uma rede pré-treinada no conjunto de dados **ImageNet**. Esse conjunto de dados possui aproximadamente 14 milhões de imagens, porém em sua grande maioria, essas imagens são reais (não ilustrações, como os exemplos ``` ursinho_carinhoso1.jpg ``` e ``` ursinho_carinhoso2.jpg ```).

 Como essa rede neural foi treinada utilizando em sua grande maioria de objetos e seres vivos reais o esperado é que ela não reconheça as ilustrações e sim apenas a imagem do urso em ``` urso_marrom_wikipedia.jpg ```, fazendo com que mesmo uma rede neural que possua uma alta versatilidade e usabilidade não funcione "corretamente".

 Caso essa rede neural tivesse em seu conjunto de dados de treinamento modelos de ilustrações de ursos ela possivelmente reconheceria os desenhos das imagens utilizadas como ursos, porém, como ela "encontra padrões", ainda sim ela poderia errar visto que existem muitas variações em estilos e características em ilustrações.

 ***

 Ao executarmos ``` beardetector.py ``` o resultado é o seguinte:

 - Para ``` urso_marrom_wikipedia.jpg ```:
    - Input image is: **brown bear**, bruin, Ursus arctos
    - It could also be: ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus, American black bear, black bear, Ursus americanus, Euarctos americanus, white wolf, Arctic wolf, Canis lupus tundrarum

 - Para ``` ursinho_carinhoso1.jpg ```:
    - Input image is: **face powder**
    - It could also be: pick, plectrum, plectron, loupe, jeweler's loupe, wall clock

 - Para ``` ursinho_carinhoso2.jpg ```:
    - Input image  is: **whistle**
    - It could also be: nematode, nematode worm, roundworm, neck brace, nipple

Com isso provamos o que já era esperado, a rede neural convolucional foi capaz de identificar apenas o animal urso, e não as ilustrações.