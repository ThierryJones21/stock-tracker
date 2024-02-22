import axios from 'axios';

const predictStocksAPI = async (object_example) => {
  try {
    // Use a relative path for the POST request
    const response = await axios.post('/backend/prediction-model', { data: object_example });
    return response.data;
  } catch (error) {
    throw new Error('Prediction error:', error);
  }
}

export default predictStocksAPI;
