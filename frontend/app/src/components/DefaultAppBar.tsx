import {AppBar, Toolbar, Typography} from "@material-ui/core";
import React from "react";
import {makeStyles} from "@material-ui/core/styles";

interface DefaultAppBarProps {
    organizationName: string;
    appName: string;
}

const useStyles = makeStyles({
    appBar: {
        height: '8%',
        width: '100%'
    },
    toolbar: {
        height: '100%'
    },
    heading: {
        height: '100%'
    }
});

export const DefaultAppBar: React.FC<DefaultAppBarProps> = ({organizationName, appName}) => {
    const classes = useStyles();

    return (
        <AppBar className={classes.appBar} position={"static"}>
            <Toolbar className={classes.toolbar} variant="dense">
                <Typography className={classes.heading}>
                    {organizationName} - {appName}
                </Typography>
            </Toolbar>
        </AppBar>
    )
}
