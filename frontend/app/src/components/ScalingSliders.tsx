import React from "react";
import {Slider} from "@material-ui/core";
import {makeStyles} from "@material-ui/core/styles";
import {projectData, updateScaling, selectDataScaling} from "../slices/dataSlice";
import {connect} from "react-redux";
import {AppDispatch, RootState} from "../app/store";


const useStyles = makeStyles({
    formItemDiv: {
        height: "100%",
        margin: "auto",
        textAlign: "center"
    }
});

interface ScalingSlidersProps {
    readonly scaling: number[],
    readonly updateScaling: any,
    readonly projectData: any
}

const ScalingSliders: React.FC<ScalingSlidersProps> = (props) => {
    const classes = useStyles();
    const scaling = props.scaling;
    const updateScaling = props.updateScaling;
    const projectData = props.projectData;

    return (
        <div className={classes.formItemDiv}>
            {Object.entries(scaling).map(
                ([key, value]) => {
                    return <Slider
                        key={`slider_${key}`}
                        orientation="vertical"
                        value={value}
                        onChange={(_, value) => {
                            if (typeof value === "number") {
                                const newScaling = [...scaling];
                                newScaling[+key] = value;
                                updateScaling(newScaling);
                                projectData();
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

const mapStateToProps = (state: RootState) => ({
    scaling: selectDataScaling(state),
});

const mapDispatchToProps = (dispatch: AppDispatch) => ({
    updateScaling: (scaling: number[]) => dispatch(updateScaling(scaling)),
    projectData: () => dispatch(projectData())
});

export default connect(
    mapStateToProps,
    mapDispatchToProps,
)(ScalingSliders);
