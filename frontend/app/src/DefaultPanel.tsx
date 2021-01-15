import {makeStyles} from "@material-ui/core/styles";
import {useDispatch} from "react-redux";
import React, {useEffect} from "react";
import {initApp, modelRemoved} from "./slices/dataSlice";
import {DefaultAppBar} from "./components/DefaultAppBar";
import {Grid} from "@material-ui/core";
import ResponsiveScatterChart from "./components/charts/ScatterChart";
import ResponsiveScatterPlotMatrixChart from "./components/charts/ScatterPlotMatrixChart";
import ResponsiveParallelCoordinatesChart from "./components/charts/ParallelCoordinatesChart";
import ScalingSliders from "./components/ScalingSliders";
import DatasetPicker from "./components/DatasetPicker";
import ProjectionPicker from "./components/ProjectionPicker";


const useStyles = makeStyles({
    app: {
        height: '96vh',
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

export const DefaultPanel: React.FC = (props) => {

    const classes = useStyles();
    const dispatch = useDispatch();

    useEffect(() => {
        dispatch(initApp());
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
                    <Grid item xs={12} sm={8} className={classes.bottomGridContainer}>
                        <ScalingSliders/>
                    </Grid>
                    <Grid container item xs={12} sm={4}>
                        <Grid item xs={12} sm={2} className={classes.controls}>
                            <DatasetPicker/>
                        </Grid>
                        <Grid item xs={12} sm={2} className={classes.controls}>
                            <ProjectionPicker/>
                        </Grid>
                    </Grid>
                </Grid>
            </div>
        </>
    );
}
