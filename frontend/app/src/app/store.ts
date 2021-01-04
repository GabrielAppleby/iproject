import {Action, configureStore, ThunkAction} from '@reduxjs/toolkit';
import dataReducer from '../slices/dataSlice';
import {TypedUseSelectorHook, useSelector} from "react-redux";


export const store = configureStore({
    reducer: {
        data: dataReducer
    },
    middleware: (getDefaultMiddleware) =>
        getDefaultMiddleware({
            serializableCheck: {ignoredActions: ['data/fetchModel/fulfilled'], ignoredPaths: ['data.model']},
            immutableCheck: {ignoredPaths: ['data.model']},
        }),
})

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
export type AppThunk<ReturnType = void> = ThunkAction<ReturnType,
    RootState,
    unknown,
    Action<string>>;

export const useTypedSelector: TypedUseSelectorHook<RootState> = useSelector;
