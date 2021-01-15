import React from 'react';
import {DefaultPanel} from "./DefaultPanel";
import ErrorDisplay from "./components/ErrorDisplay";


function App() {

    return (
        <ErrorDisplay>
            <DefaultPanel/>
        </ErrorDisplay>
    );
}

export default App;
