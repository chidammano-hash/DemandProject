// Side-effect imports — each calls registerMotif() at module evaluation time
import "./periodicMotif";
import "./spiritsMotif";
import "./spaceMotif";
import "./f1Motif";
import "./zenMotif";

export { getMotif, getAllMotifs, DEFAULT_MOTIF_ID, registerMotif } from "@/constants/motifRegistry";
