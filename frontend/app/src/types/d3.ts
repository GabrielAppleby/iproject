import * as d3 from "d3";

export type RootSelection = d3.Selection<SVGSVGElement, unknown, null, undefined>;
export type GroupSelection = d3.Selection<Element | d3.EnterElement | Document | Window | SVGGElement | null, number[], d3.BaseType, unknown>
