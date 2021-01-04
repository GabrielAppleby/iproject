import {createAsyncThunk, createEntityAdapter, createSlice, EntityState,} from '@reduxjs/toolkit';
import {AppDispatch, RootState} from '../app/store';
import {getData} from '../api/dataClient'
import {Data, DataInstance, Dataset, Status} from "../types/data";
import * as tf from "@tensorflow/tfjs";
import {getModel} from "../api/modelClient";
import {getDatasetEndpoint} from "../api/common";


interface DataState extends EntityState<DataInstance> {
    name: Dataset;
    scaling: number[];
    model?: tf.LayersModel;
    dataStatus: Status;
    modelStatus: Status;
    projectionStatus: Status;
}


const dataAdapter = createEntityAdapter<DataInstance>({
    selectId: instance => instance.uid
});

const initialState = dataAdapter.getInitialState({
    name: "iris",
    scaling: [1, 1, 1, 1],
    dataStatus: 'idle',
    modelStatus: 'idle',
    projectionStatus: 'idle',
    model: undefined
}) as DataState;


export const fetchData = createAsyncThunk<Data, void, {state: RootState}>('data/fetchData', async (arg, thunkAPI) => {
    const endPoint = getDatasetEndpoint(thunkAPI.getState().data.name) + 's';

    return await getData<Data>(endPoint);
});

export const fetchModel = createAsyncThunk<tf.LayersModel, void, {state: RootState}>('data/fetchModel', async (arg, thunkAPI) => {
    const endPoint = getDatasetEndpoint(thunkAPI.getState().data.name) + '';

    return await getModel(endPoint);
});

type ProjectionChanges = { changes: { projection: number[]; }; id: number; }[];

export const projectData = createAsyncThunk<ProjectionChanges,
    void,
    {
        dispatch: AppDispatch,
        state: RootState,
        rejectValue: string
    }>('data/projectData', async (arg, thunkAPI) => {
    const state = thunkAPI.getState();
    const model = selectModel(state);
    if (model !== undefined) {
        const data = dataSelecters.selectAll(state);
        const scaling = selectDataScaling(state);

        const featureArray = data.map((d) => [...d.features].concat(scaling));
        const ids = data.map((d) => d.uid);
        const predictions = tf.tidy(() => {
            const features = tf.tensor(featureArray);
            return (model.predict(features) as tf.Tensor<tf.Rank.R2>).arraySync();
        });
        return predictions.map((d, i) => {
            return {'changes': {'projection': d}, id: ids[i]}
        });
    } else {
        return thunkAPI.rejectWithValue("Model not defined");
    }
}, {condition: (arg, api) => api.getState().data.projectionStatus !== 'pending'})

export const dataSlice = createSlice({
    name: 'data',
    initialState,
    reducers: {
        changeDataset(state, action) {
            state.name = action.payload;
        },
        updateScaling(state, action) {
            state.scaling = action.payload;
        },
        modelRemoved(state) {
            if (state.model !== undefined) {
                state.model.dispose();
            }
        }
    },
    extraReducers: builder => {
        builder.addCase(fetchData.pending, (state, action) => {
            state.dataStatus = 'pending'
        })
        builder.addCase(fetchData.fulfilled, (state, action) => {
            state.dataStatus = 'fulfilled'
            dataAdapter.removeAll(state);
            dataAdapter.upsertMany(state, action.payload)
        })
        builder.addCase(fetchData.rejected, (state, action) => {
            state.dataStatus = 'rejected'
        })
        builder.addCase(fetchModel.pending, (state, action) => {
            state.modelStatus = 'pending'
        })
        builder.addCase(fetchModel.fulfilled, (state, action) => {
            state.modelStatus = 'fulfilled'
            state.model = action.payload;
        })
        builder.addCase(fetchModel.rejected, (state, action) => {
            state.modelStatus = 'rejected'
        })
        builder.addCase(projectData.pending, (state, action) => {
            state.projectionStatus = 'pending'
        })
        builder.addCase(projectData.fulfilled, (state, action) => {
            state.projectionStatus = 'fulfilled'
            if (action.payload !== undefined) {
                dataAdapter.updateMany(state, action.payload)
            }
        })
        builder.addCase(projectData.rejected, (state, action) => {
            state.projectionStatus = 'rejected'
        })
    }
});

export const {updateScaling, modelRemoved, changeDataset} = dataSlice.actions;

const dataSelecters = dataAdapter.getSelectors<RootState>(state => state.data);

export const selectAllData = (state: RootState) => {
    return dataSelecters.selectAll(state);
}

export const selectDataScaling = (state: RootState) => {
    return state.data.scaling;
}

export const selectDataset = (state: RootState) => {
    return state.data.name;
}

export const selectModel = (state: RootState) => {
    return state.data.model;
}

export const selectAllScaledData = (state: RootState) => {
    const scaling = selectDataScaling(state);
    return selectAllData(state).map(instance => {
        return {
            uid: instance.uid,
            target: instance.target,
            features: instance.features.map((v, i) => v * scaling[i])
        }
    })
}

export default dataSlice.reducer;
