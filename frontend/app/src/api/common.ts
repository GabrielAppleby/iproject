import {Dataset} from "../types/data";

export const API_URL: string = process.env.REACT_APP_API || '';

export const getDatasetEndpoint = (name: Dataset) => {
    if (name === 'iris')
    {
        return '/flower'
    }
    else if (name === 'wine')
    {
        return '/wine'
    }
    else if (name === 'ecoli')
    {
        return '/bacteria'
    }

}
