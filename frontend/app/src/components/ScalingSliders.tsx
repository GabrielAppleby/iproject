import React from "react";
import {Slider} from "@material-ui/core";
import {makeStyles} from "@material-ui/core/styles";
import {selectDataScaling, updateScalingsAndProject} from "../slices/dataSlice";
import {connect, ConnectedProps} from "react-redux";
import {AppDispatch, RootState} from "../app/store";


const useStyles = makeStyles({
    formItemDiv: {
        height: "100%",
        margin: "auto",
        // textAlign: "center"
    },
    sliderDiv: {
        marginRight: "10%",
        marginLeft: "10%"
    }
});

const mapStateToProps = (state: RootState) => ({
    scaling: selectDataScaling(state),
});

const mapDispatchToProps = (dispatch: AppDispatch) => ({
    updateScalingAndProject: (scaling: number[]) => dispatch(updateScalingsAndProject(scaling))
});

const connector = connect(
    mapStateToProps,
    mapDispatchToProps,
);

type PropsFromRedux = ConnectedProps<typeof connector>

const ScalingSliders: React.FC<PropsFromRedux> = (props) => {
    const classes = useStyles();
    const scaling = props.scaling;
    const updateScalingAndProject = props.updateScalingAndProject;

    return (
        <div className={classes.formItemDiv}>
            {Object.entries(scaling).map(
                ([key, value]) => {
                    return <Slider
                        className={classes.sliderDiv}
                        key={`slider_${key}`}
                        orientation="vertical"
                        value={value}
                        onChange={(_, value) => {
                            if (typeof value === "number") {
                                const newScaling = [...scaling];
                                newScaling[+key] = value;
                                updateScalingAndProject(newScaling);
                            }
                        }}
                        valueLabelDisplay="auto"
                        min={0.0}
                        max={1.0}
                        step={.01}
                    />
                }
            )}
        </div>
    );
}

export default connector(ScalingSliders);
