import React from "react";
import {Select} from "@material-ui/core";
import {changeDatasetAndFetchData, selectDataset} from "../slices/dataSlice";
import {connect, ConnectedProps} from "react-redux";
import {AppDispatch, RootState} from "../app/store";
import {Dataset} from "../types/data";

const mapStateToProps = (state: RootState) => ({
    dataset: selectDataset(state),
});

const mapDispatchToProps = (dispatch: AppDispatch) => ({
    changeDataset: (name: Dataset) => dispatch(changeDatasetAndFetchData(name)),
});

const connector = connect(
    mapStateToProps,
    mapDispatchToProps,
);

type PropsFromRedux = ConnectedProps<typeof connector>


const DatasetPicker = (props: PropsFromRedux) => {
    const selected = props.dataset;
    const changeSelection = props.changeDataset;


    return (
        <Select
            native
            value={selected}
            onChange={(event: React.ChangeEvent<{ value: unknown }>) => {
                changeSelection(event.target.value as Dataset);
            }}
            inputProps={{
                name: 'dataset',
                id: 'dataset-native-simple',
            }}>
            <option value={'iris'}>Iris</option>
        </Select>
    );
}

export default connector(DatasetPicker);
