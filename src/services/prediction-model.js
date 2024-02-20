// import { sequential, layers as _layers } from '@tensorflow/tfjs-node';


// export const predictStocks = async (trainingData)=>{
   
//     const model = sequential({
//     layers: [
//         _layers.dense({ inputShape: [1], units: 64, activation: 'relu' }),
//         _layers.dense({ units: 1 }),
//     ],
//     });

//     model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

//     const xs = tf.tensor1d(trainingData.map(item => item.x));
//     const ys = tf.tensor1d(trainingData.map(item => item.y));

//     await model.fit(xs, ys, { epochs: 50 });

//     const futureDates = 3
//     const futureTensor = tf.tensor1d(futureDates);
//     const predictions = model.predict(futureTensor).dataSync();

//     return predictions
// }