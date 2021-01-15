import React, {useEffect, useRef} from "react";
import {RootState} from "../../app/store";
import {selectAllScaledData} from "../../slices/dataSlice";
import {Dimensions, withDimensions} from "../../wrappers/Dimensions";
import * as d3 from "d3";
import {Data} from "../../types/data";
import {COLORS, getFixedDomainExtents} from "./common";
import {GroupSelection} from "../../types/d3";
import {connect} from "react-redux";

type RootSelection = d3.Selection<SVGSVGElement, unknown, null, undefined>;

interface Size {
    readonly w: number
    readonly h: number
}

interface SPLOMScales {
    readonly xScales: d3.ScaleLinear<number, number>[];
    readonly yScales: d3.ScaleLinear<number, number>[];
}

const BUFFER_PROPORTION = 1 / 20;


const createGroups = (rootG: RootSelection) => {
    rootG.append("g").attr("id", "yAxisG");
    rootG.append("g").attr("id", "xAxisG");
    rootG.append("g").attr("id", "cellG");
}

const getMargins = ({width, height}: Dimensions) => {
    return {w: width * (1 / 40), h: height * (1 / 40)};
}

const getPlotSizes = ({width, height}: Dimensions, margins: Size, numFeatures: number) => {
    const w = (width - (numFeatures + 1) * margins.w) / numFeatures
    const h = (height - (numFeatures + 1) * margins.h) / numFeatures

    return {w, h}
}

const getRangeExtents = (data: Data, domainExtents: number[][]) => {
    return domainExtents.map(extrema => {
        const range = extrema[1] - extrema[0];
        return [extrema[0] - (range * BUFFER_PROPORTION), extrema[1] + (range * BUFFER_PROPORTION)]
    });
}

const joinPlotGroups = (rootG: RootSelection, plotCrossProducts: number[][], plotSizes: Size, margins: Size) => {
    return rootG.select('#cellG')
        .selectAll("g")
        .data(plotCrossProducts)
        .join("g")
        .attr("transform", ([i, j]) => {
            return `translate(${(i * (plotSizes.w + margins.w)) + margins.w},${(j * (plotSizes.h + margins.h)) + margins.h})`
        });
}

const getScales = (numFeatures: number, extents: number[][], plotSizes: Size) => {
    const x = d3.range(numFeatures).map(c => d3.scaleLinear()
        .domain(extents[c])
        .range([0, plotSizes.w]));

    const y = x.map((x, i) => x.copy().range([plotSizes.h, 0]));
    return {xScales: x, yScales: y}
}

const joinRects = (cell: GroupSelection, plotCrossProducts: number[][], plotSizes: Size) => {
    cell.each(function ([i, j]) {
        d3.select(this)
            .selectAll("rect")
            .data(plotCrossProducts)
            .join("rect")
            .attr("fill", "none")
            .attr("stroke-width", 1.5)
            .attr("stroke", "#aaa")
            .attr("width", plotSizes.w)
            .attr("height", plotSizes.h);
    });
}

const joinCircles = (cell: GroupSelection, data: Data, scales: SPLOMScales) => {
    cell.each(function ([i, j]) {
        d3.select(this)
            .selectAll("circle")
            .data(data)
            .join("circle")
            .attr("cx", d => scales.xScales[i](d.features[i]))
            .attr("cy", d => scales.yScales[j](d.features[j]))
            .attr("r", 1.5)
            .attr("fill", d => COLORS(d.target))
            .attr("fill-opacity", 0.7);
    });
}


interface ScatterPlotMatrixChartProps {
    readonly dimensions: Dimensions;
    readonly data: Data;
}

const ScatterPlotMatrixChart: React.FC<ScatterPlotMatrixChartProps> = (props) => {
    const d3Container = useRef<SVGSVGElement>(null);
    // const data = useTypedSelector((state) => selectAllScaledData(state));
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
            const margins = getMargins(dimensions);
            const plotSizes = getPlotSizes(dimensions, margins, numFeatures);
            const domainExtents = getFixedDomainExtents(numFeatures);
            const extents = getRangeExtents(data, domainExtents)
            const plotCrossProducts = d3.cross(d3.range(numFeatures), d3.range(numFeatures));
            const rootG = d3.select(d3Container.current);
            const cell = joinPlotGroups(rootG, plotCrossProducts, plotSizes, margins);
            const scales: SPLOMScales = getScales(numFeatures, extents, plotSizes);
            joinRects(cell, plotCrossProducts, plotSizes);
            joinCircles(cell, data, scales);

        }
    }, [data, dimensions]);

    return (<svg ref={d3Container} width={dimensions.width} height={dimensions.height}/>)
}


const ResponsiveScatterPlotMatrixChart = withDimensions(ScatterPlotMatrixChart);

const mapStateToProps = (state: RootState) => ({
    data: selectAllScaledData(state),
});

export default connect(
    mapStateToProps,
)(ResponsiveScatterPlotMatrixChart);
