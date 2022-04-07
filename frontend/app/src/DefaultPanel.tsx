import {makeStyles} from "@material-ui/core/styles";
import {useDispatch} from "react-redux";
import React, {useEffect} from "react";
import {initApp, modelRemoved} from "./slices/dataSlice";
import {DefaultAppBar} from "./components/DefaultAppBar";
import ResponsiveScatterChart from "./components/charts/ScatterChart";
import ResponsiveScatterPlotMatrixChart from "./components/charts/ScatterPlotMatrixChart";
import ResponsiveParallelCoordinatesChart from "./components/charts/ParallelCoordinatesChart";
import ScalingSliders from "./components/ScalingSliders";

const useStyles = makeStyles({
    app: {
        height: '96vh',
        width: '100vw'
    },
    mainGrid: {
        overflow: 'hidden',
        height: '92%',
        width: '100%'
    },
    chartTopContainer: {
        height: '80%',
        width: '100%'
    },
    chartBottomContainer: {
        marginTop: '2%',
        marginBottom: '2%',
        height: '16%',
        width: '100%'
    },
    chartTopInnerContainer: {
        height: '100%',
        width: '50%',
        float: 'left'
    },
    chartTopInnerRightContainer: {
        height: '50%',
        width: '100%',
        float: 'left'
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
                {/*<DefaultAppBar organizationName={"VALT"} appName={"iProject"}/>*/}
                <div className={classes.mainGrid}>
                    <div className={classes.chartTopContainer}>
                        <div className={classes.chartTopInnerContainer}>
                            <ResponsiveScatterChart/>
                        </div>
                        <div className={classes.chartTopInnerContainer}>
                            <div className={classes.chartTopInnerRightContainer}>
                                <ResponsiveScatterPlotMatrixChart/>
                            </div>
                            <div className={classes.chartTopInnerRightContainer}>
                                <ResponsiveParallelCoordinatesChart/>
                            </div>
                        </div>
                    </div>
                    <div className={classes.chartBottomContainer}>
                        <ScalingSliders/>
                    </div>
                </div>
            </div>
        </>
    );
}
