export type NumberTuple = [number, number];
export type UndefinedTuple = [undefined, undefined];
export type UnknownTuple = NumberTuple | UndefinedTuple;
export type UnknownTuples = UnknownTuple[];
export type NumberTuples = NumberTuple[];

export interface ExtendedExtents {
    readonly startX: number;
    readonly endX: number;
    readonly startY: number;
    readonly endY: number;
}
