<template>
  <v-row>
    <NavDrawer />
    <v-dialog v-model="dialog" persistent max-width="300px">
      <v-card>
        <v-card-title>
          <span class="text-h5">{{
            `Add ${nodeCategory[nodeType]} layer`
          }}</span>
        </v-card-title>
        <v-card-text>
          <v-container>
            <v-form ref="form">
              <v-row>
                <v-col cols="12">
                  <v-text-field
                    v-model="nodeLabel"
                    label="Name"
                    :prepend-icon="nodeIcons[nodeType]"
                    :rules="[rules.length(3)]"
                  ></v-text-field>
                </v-col>
              </v-row>
            </v-form>
          </v-container>
        </v-card-text>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn color="grey darken-1" text @click="cancel"> Cancel </v-btn>
          <v-btn color="blue darken-1" text @click="add"> add </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
    <v-col cols="12" class="pa-0 primary">
      <div class="d-flex flex-column float-left pt-5">
        <v-btn dark plain small @click="dialog = true" color="text">
          <v-icon>mdi-plus</v-icon>
        </v-btn>
        <v-btn small plain text :color="isColor(0)" @click="nodeType = 0">
          <v-icon>mdi-format-horizontal-align-left</v-icon>
        </v-btn>
        <v-btn small plain text :color="isColor(1)" @click="nodeType = 1">
          <v-icon>mdi-format-horizontal-align-center</v-icon>
        </v-btn>
        <v-btn small plain text :color="isColor(2)" @click="nodeType = 2">
          <v-icon>mdi-format-horizontal-align-right</v-icon>
        </v-btn>
        <v-btn dark plain small disabled @click="save" color="text">
          <v-icon>mdi-cloud-download-outline</v-icon>
        </v-btn>
        <v-btn dark plain small @click="save" color="text">
          <v-icon>mdi-cloud-upload-outline</v-icon>
        </v-btn>
      </div>
      <div>
        <simple-flowchart
          class="primary lighten-1"
          :scene.sync="scene"
          @nodeClick="nodeClick"
          @nodeDelete="nodeDelete"
          @linkBreak="linkBreak"
          @linkAdded="linkAdded"
          @canvasClick="canvasClick"
          :height="800"
        />
      </div>
    </v-col>
  </v-row>
</template>

<script>
import SimpleFlowchart from "@/components/flowchart/SimpleFlowchart";
import NavDrawer from "@/components/app/NavDrawer";
import { mapGetters } from "vuex";

export default {
  name: "app",
  components: {
    SimpleFlowchart,
    NavDrawer,
  },
  data() {
    return {
      dialog: false,
      nodeType: 1,
      nodeLabel: "",
      nodeCategory: ["input", "action", "output"],
      nodeIcons: [
        "mdi-format-horizontal-align-left",
        "mdi-format-horizontal-align-center",
        "mdi-format-horizontal-align-right",
      ],
      rules: {
        length: (len) => (v) => (v || "").length >= len || `Length < ${len}`,
      },
    };
  },
  computed: {
    ...mapGetters({
      scene: "data/getData",
    }),
    drawer: {
      set(value) {
        this.$store.dispatch("settings/setDrawer", value);
      },
      get() {
        return this.$store.getters["settings/getDrawer"];
      },
    },
  },
  methods: {
    canvasClick(e) {
      console.log("canvas Click, event:", e);
      console.log(e.type)
    },
    add() {
      if (this.$refs.form.validate()) {
        let maxID = Math.max(
          0,
          ...this.scene.nodes.map((link) => {
            return link.id;
          })
        );
        this.scene.nodes.push({
          id: maxID + 1,
          x: -400,
          y: -100,
          type: this.nodeCategory[this.nodeType],
          label: this.nodeLabel ? this.nodeLabel : `test${maxID + 1}`,
        });
        this.nodeLabel = "";
        this.dialog = false;
      }
    },
    cancel() {
      console.log(this.$refs.form.reset());
      this.nodeLabel = "";
      this.dialog = false;
    },
    save() {
      const { nodes, links } = this.scene;
      console.log({ nodes, links });
      alert(JSON.stringify({ nodes, links }));
    },
    nodeClick(id) {
      console.log("node click", id);
      // this.drawer = true
    },
    nodeDelete(id) {
      console.log("node delete", id);
    },
    linkBreak(id) {
      console.log("link break", id);
    },
    linkAdded(link) {
      console.log("new link added:", link);
    },
    isColor(type) {
      return this.nodeType !== type ? "text" : "white";
    },
  },
};
</script>

<style lang="scss" scoped>
.sidebar {
  float: left;
  width: auto;
}
</style>
