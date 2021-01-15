import {NumberTuples, UnknownTuples} from "../../types/charts";
import {Data, ProjectedData} from "../../types/data";
import * as d3 from "d3";

export const DOMAIN_EXTENT = [0, 1];
export const COLORS = d3.scaleOrdinal(d3.schemeCategory10);

export const getFixedDomainExtents = (numFeatures: number) => {
    return d3.range(numFeatures).map((i) => {
        return DOMAIN_EXTENT;
    });
}

export function isNumberTuples(extents: UnknownTuples): extents is NumberTuples {
    return isNotNullish(extents);
}

export function isNotNullish<T>(value: T): value is NonNullable<T> {
    if (Array.isArray(value)) {
        return value.every(isNotNullish);
    }
    return value !== undefined && value !== null;
}

export const isProjected = (x: Data): x is ProjectedData => {
    return x[0] !== undefined && x[0].projections !== undefined;
}
