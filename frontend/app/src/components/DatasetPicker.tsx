import React from "react";
import {Select} from "@material-ui/core";
import {selectDataset, changeDataset, fetchData} from "../slices/dataSlice";
import {connect} from "react-redux";
import {AppDispatch, RootState} from "../app/store";
import {Dataset} from "../types/data";


interface DatasetPickerProps {
    readonly dataset: string,
    readonly changeDataset: any,
    readonly fetchData: any,
}

const DatasetPicker: React.FC<DatasetPickerProps> = (props) => {
    const dataset = props.dataset;
    const changeDataset = props.changeDataset;
    const fetchData = props.fetchData;

    return (
        <Select
            native
            value={dataset}
            onChange={(_, value) => {
                changeDataset(value);
                fetchData(value);
            }}
            inputProps={{
                name: 'dataset',
                id: 'dataset-native-simple',
            }}>
            <option value={"iris"}>Iris</option>
            <option value={"wine"}>Wine</option>
            <option value={"ecoli"}>Ecoli</option>
        </Select>
    );
}

const mapStateToProps = (state: RootState) => ({
    dataset: selectDataset(state),
});

const mapDispatchToProps = (dispatch: AppDispatch) => ({
    changeDataset: (name: Dataset) => dispatch(changeDataset(name)),
    fetchData: () => dispatch(fetchData())
});

export default connect(
    mapStateToProps,
    mapDispatchToProps,
)(DatasetPicker);
