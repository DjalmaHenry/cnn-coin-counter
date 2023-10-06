# cnn-coin-counter
Um sistema de reconhecimento e soma de moedas em tempo real.

## 📷 Reconhecimento e Contagem

<p align="center">
  <img alt="Imagem da identificação e contagem" src="https://github.com/DjalmaHenry/cnn-coin-counter/assets/63603061/d921e670-7b2a-40e9-b754-eb729a5cd007" width="50%">
</p>

## 💾 Base de dados
Banco com imagens de moedas de tamanho 224x224:
1. 1 Real (212 imagens)
2. 0.50 Centavos (198 imagens)
3. 0.25 Centavos (244 imagens)

## 🔬 Preprocessamento
- Gaussianblur: expandir pixels da imagem, deixando-a borrada
- Canny: Filtrar imagens pelas bordas
- Diltação
- Erosão

## 📏 Métrica de avaliação
Percentual de confiança

## 🕸️ 3 modelos CNN:
- 2 de profundidade
- 4 de profundidade
- 6 de profundidade

## 👨‍👦‍👦 Colaborares
#### Nome: Djalma Henrique Silva Lima
- GitHub: [DjalmaHenry](https://github.com/DjalmaHenry)
#### Nome: Ronny Lima Ribeiro da Silva
- GitHub: [ronnylrsd](https://github.com/ronnylrsd)
