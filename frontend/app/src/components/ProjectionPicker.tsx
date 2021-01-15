import React from "react";
import {Select} from "@material-ui/core";
import {changeProjectionAndFetchModel, selectProjection} from "../slices/dataSlice";
import {connect, ConnectedProps} from "react-redux";
import {AppDispatch, RootState} from "../app/store";
import {Projection} from "../types/data";

const mapStateToProps = (state: RootState) => ({
    projection: selectProjection(state),
});

const mapDispatchToProps = (dispatch: AppDispatch) => ({
    changeProjectionAndFetchModel: (projection: Projection) => dispatch(changeProjectionAndFetchModel(projection)),
});

const connector = connect(
    mapStateToProps,
    mapDispatchToProps,
);

type PropsFromRedux = ConnectedProps<typeof connector>


const ProjectionPicker = (props: PropsFromRedux) => {
    const selected = props.projection;
    const changeSelection = props.changeProjectionAndFetchModel;


    return (
        <Select
            native
            value={selected}
            onChange={(event: React.ChangeEvent<{ value: unknown }>) => {
                changeSelection(event.target.value as Projection);
            }}
            inputProps={{
                name: 'projection',
                id: 'projection-native-simple',
            }}>
            <option value={'umap'}>umap</option>
        </Select>
    );
}

export default connector(ProjectionPicker);
