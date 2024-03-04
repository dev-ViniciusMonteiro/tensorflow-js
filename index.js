const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

// Frases de exemplo para treinamento
const trainingData = [
    { input: 'Quero marcar um horário para amanhã.', output: 'agendar' },
    { input: 'Gostaria de mudar meu horário marcado.', output: 'mudar' },
    { input: 'Quais são meus horários disponíveis?', output: 'consultar' }
    // Adicione mais exemplos conforme necessário
];

// Converta os dados de treinamento para tensores
const trainXs = tf.tensor2d(trainingData.map(item => item.input.split('').map(char => char.charCodeAt(0))));
const trainYs = tf.tensor2d(trainingData.map(item => [item.output === 'agendar' ? 1 : 0, item.output === 'mudar' ? 1 : 0, item.output === 'consultar' ? 1 : 0]));

// Defina o modelo de rede neural
const model = tf.sequential();
model.add(tf.layers.dense({ units: 8, inputShape: [trainXs.shape[1]], activation: 'relu' }));
model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

// Compile o modelo
model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

// Treine o modelo
model.fit(trainXs, trainYs, { epochs: 100 }).then(() => {
    console.log('Treinamento concluído.');

    // Exemplo de uso
    const novaFrase = 'Quero marcar um horário para hoje à tarde.';
    const tokens = novaFrase.split('').map(char => char.charCodeAt(0));
    const inputTensor = tf.tensor2d([tokens]);
    const outputTensor = model.predict(inputTensor);
    const outputData = outputTensor.dataSync();
    const intencaoIndex = outputData.indexOf(Math.max(...outputData));

    let intencao;
    if (intencaoIndex === 0) {
        intencao = 'agendar';
    } else if (intencaoIndex === 1) {
        intencao = 'mudar';
    } else {
        intencao = 'consultar';
    }

    console.log('Intenção:', intencao);
});
