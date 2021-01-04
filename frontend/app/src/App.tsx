import React, {useEffect} from 'react';
import {makeStyles} from "@material-ui/core/styles";
import {DefaultAppBar} from "./components/DefaultAppBar";
import {Grid} from "@material-ui/core";
import {useDispatch} from "react-redux";
import {fetchData, fetchModel, modelRemoved, projectData} from "./slices/dataSlice";
import ResponsiveParallelCoordinatesChart from "./components/charts/ParallelCoordinatesChart";
import ResponsiveScatterPlotMatrixChart from "./components/charts/ScatterPlotMatrixChart";
import ResponsiveScatterChart from "./components/charts/ScatterChart";
import ScalingSliders from "./components/ScalingSliders";
import DatasetPicker from "./components/DatasetPicker";


const useStyles = makeStyles({
    app: {
        height: '98vh',
        display: 'flex',
        flexDirection: 'column'
    },
    mainGrid: {
        height: '1%',
        flexGrow: 1
    },
    chartGridContainer: {
        height: '40%'
    },
    bottomGridContainer: {
        height: '20%'
    },
    controls: {
        margin: 'auto',
        textAlign: 'center'
    }
});

function App() {

    const classes = useStyles();
    const dispatch = useDispatch();

    useEffect(() => {
        dispatch(fetchData());
        dispatch(fetchModel());
    });

    useEffect(() => {
        return () => {
            dispatch(modelRemoved())
        };
    });

    return (
        <>
            <div className={classes.app}>
                <DefaultAppBar organizationName={"VALT"} appName={"iProject"}/>
                <Grid container item className={classes.mainGrid}>
                    <Grid item xs={12} md={6} className={classes.chartGridContainer}>
                        <ResponsiveScatterChart/>
                    </Grid>
                    <Grid item xs={12} md={6} className={classes.chartGridContainer}>
                        <ResponsiveScatterPlotMatrixChart/>
                    </Grid>
                    <Grid item xs={12} className={classes.chartGridContainer}>
                        <ResponsiveParallelCoordinatesChart/>
                    </Grid>
                    <Grid item xs={12} sm={10} className={classes.bottomGridContainer}>
                        <ScalingSliders/>
                    </Grid>
                    <Grid item xs={12} sm={2} className={classes.controls}>
                        <DatasetPicker/>
                    </Grid>
                </Grid>
            </div>
        </>
    );
}

export default App;
