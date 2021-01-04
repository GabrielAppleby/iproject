import React, {useEffect, useRef} from "react";

import * as d3 from "d3";
import {Dimensions, withDimensions} from "../../wrappers/Dimensions";
import {selectAllScaledData} from "../../slices/dataSlice";
import {RootState} from "../../app/store";
import {Data, DataInstance} from "../../types/data";
import {ExtendedExtents} from "../../types/charts";
import {RootSelection} from "../../types/d3";
import {COLORS, getFixedDomainExtents} from "./common";
import {connect} from "react-redux";

// The version of d3 types is like two major versions behind the actual d3
// version, so for the most part ignore typescript

interface ParallelCoordinatesScales {
    readonly xScale: d3.ScalePoint<string>;
    readonly yScales: d3.ScaleLinear<number, number>[];
}

const createGroups = (rootG: RootSelection) => {
    rootG.append("g").attr("id", "pathsG");
    rootG.append("g").attr("id", "axisG");
}


const getScales = (domainExtents: number[][], {startX, endX, startY, endY}: ExtendedExtents) => {
    const y: d3.ScaleLinear<number, number>[] = [];

    domainExtents.forEach((extrema, i) => {
        y[i] = d3.scaleLinear()
            .domain(extrema)
            .range([startY, endY]);
    })

    const x = d3.scalePoint()
        .range([startX, endX])
        .padding(0)
        .domain(Object.keys(y));

    return {xScale: x, yScales: y};
}


const callAxis = (rootG: RootSelection,
                  {xScale, yScales}: ParallelCoordinatesScales) => {
    const axisG = rootG.select('#axisG');
    axisG.selectAll(".axis")
        .data(yScales)
        .join(enter => enter.append('g').attr("class", "axis"))
        .attr("transform", function (d, i) {
            return "translate(" + xScale(i.toString()) + ")";
        })
        .each(function (d, i) {
            // @ts-ignore
            d3.select(this).call(d3.axisLeft(d).tickFormat(() => "").tickSize(0));
        });
}


const joinPaths = (rootG: RootSelection,
                   data: Data,
                   {xScale, yScales}: ParallelCoordinatesScales) => {
    const pathsG = rootG.select('#pathsG');

    function path(d: DataInstance) {
        // @ts-ignore
        return d3.line()(yScales.map(function (sl, i) {
            return [xScale(i.toString()), sl(d.features[i])]
        }));
    }

    pathsG
        .selectAll("path")
        .data(data)
        .join("path")
        .attr("d", path)
        .style("fill", "none")
        .style("stroke", d => COLORS(d.target))
        .style("opacity", 0.5);
}


const getRangeExtents = (dimensions: Dimensions) => {
    const {width, height} = dimensions;
    const margins = {w: width * (1 / 40), h: height * (1 / 40)};
    return {
        startX: margins.w,
        endX: (width - margins.w),
        startY: (height - margins.h),
        endY: margins.h
    }
}

interface ParallelCoordinatesChartProps {
    readonly dimensions: Dimensions;
    readonly data: Data;
}

const ParallelCoordinatesChart: React.FC<ParallelCoordinatesChartProps> = (props) => {
    const d3Container = useRef<SVGSVGElement>(null);
    const data = props.data;
    const dimensions = props.dimensions;

    useEffect(() => {
        if (d3Container.current !== null) {
            const rootG = d3.select(d3Container.current);
            createGroups(rootG);
        }
    }, []);


    useEffect(() => {
        if (data.length > 0 && d3Container.current !== null) {
            const numFeatures = data[0].features.length;
            const rangeExtents = getRangeExtents(dimensions)
            const domainExtents = getFixedDomainExtents(numFeatures);
            const scales = getScales(domainExtents, rangeExtents);
            const rootG = d3.select(d3Container.current);
            callAxis(rootG, scales);
            joinPaths(rootG, data, scales);
        }
    }, [data, dimensions]);

    return (
        <svg ref={d3Container} width={dimensions.width} height={dimensions.height}/>
    )

}

const ResponsiveParallelCoordinatesChart = withDimensions(ParallelCoordinatesChart);

const mapStateToProps = (state: RootState) => ({
    data: selectAllScaledData(state),
});

export default connect(
    mapStateToProps,
)(ResponsiveParallelCoordinatesChart);

