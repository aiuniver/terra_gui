<template>
  <div>
    <v-dialog v-model="dialog" persistent max-width="290">
      <v-card>
        <v-card-title class="text-h5">
          {{ dialogTitle }}
        </v-card-title>
        <v-card-text v-if="isLoad">
          Please stand by
          <v-progress-linear indeterminate class="mb-0"></v-progress-linear>
        </v-card-text>
        <v-card-text v-show="!isLoad">
          <v-container>
            <v-row>
              <v-form ref="form">
                <v-col cols="12">
                  <v-text-field
                    v-model="inputName"
                    label="Name"
                    type="text"
                    prepend-icon="mdi-layers-outline"
                    :rules="[rules.length(3)]"
                    :disabled="dialogTitle === 'Delete'"
                  ></v-text-field>
                </v-col>
                <v-col cols="12">
                  <v-text-field
                    v-model="inputSize"
                    label="Size"
                    type="number"
                    prepend-icon="mdi-content-save-outline"
                    :rules="[rules.required]"
                    :disabled="dialogTitle === 'Delete'"
                  ></v-text-field>
                </v-col>
                <v-col cols="12">
                  <v-select
                    v-model="inputTags"
                    :items="tagsArr"
                    :menu-props="{ maxHeight: '400' }"
                    label="Tags"
                    multiple
                    prepend-icon="mdi-tag"
                    :rules="[rules.required]"
                    persistent-hint
                    return-object
                    :disabled="dialogTitle === 'Delete'"
                  ></v-select>
                </v-col>
              </v-form>
            </v-row>
          </v-container>
        </v-card-text>
        <v-card-actions v-if="!isLoad">
          <v-spacer></v-spacer>
          <v-btn color="grey darken-1" text @click="cancel"> Cancel </v-btn>
          <v-btn color="red darken-1" text @click="click(dialogTitle)">
            {{ dialogTitle }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
    <v-row>
      <v-col cols="9">
        <v-row>
          <v-col cols="12" class="align-center pb-0 d-flex">
            <h5 class="success--text">Tags</h5>
          </v-col>
          <v-col cols="12">
            <v-chip-group v-model="tagsFilter" column multiple>
              <v-chip
                v-for="(tag, key) in tags"
                :key="key"
                class="ma-1"
                text-color="success"
                dark
                label
                :value="key"
                small
                filter
                outlined
              >
                {{ tag }}
              </v-chip>
            </v-chip-group>
          </v-col>
          <v-col cols="12" class="align-center pb-0 d-flex">
            <h5 class="success--text">Datasets</h5>
          </v-col>
          <v-col cols="12">
            <v-row>
              <v-col
                v-for="(dataset, i) in datasets"
                :key="i"
                cols="12"
                sm="6"
                md="4"
                lg="3"
                xl="2"
              >
                <Card :dataset="dataset" @change="change" />
              </v-col>
            </v-row>
          </v-col>
          <v-col cols="12">
            <h3 v-if="!datasets.length" class="text-center">Not datasets</h3>
          </v-col>
        </v-row>
      </v-col>
      <v-col cols="3" class="pa-0">
        <Settings />
      </v-col>
    </v-row>
  </div>
</template>

<script>
import { mapGetters } from "vuex";
import Card from "@/components/dataset/Card";
import Settings from "@/components/dataset/Settings";

export default {
  name: "Datasets",
  components: {
    Card,
    Settings,
  },
  data: () => ({
    modalDel: false,
    dialog: false,
    dialogTitle: "",
    inputName: "",
    inputDate: "",
    inputSize: 0,
    inputTags: [],
    isLoad: false,
    isChange: false,
    isNew: false,
    id: null,
    rules: {
      length: (len) => (v) => (v || "").length >= len || `Length < ${len}`,
      required: (len) => len.length !== 0 || `Not be empty`,
    },
  }),
  computed: {
    ...mapGetters({
      tags: "datasets/getTags",
      tagsArr: "datasets/getTagsArr",
      datasets: "datasets/getDatasets",
    }),
    tagsFilter: {
      set(value) {
        this.$store.dispatch("datasets/setTagsFilter", value);
      },
      get() {
        return this.$store.getters["datasets/getTagsFilter"];
      },
    },
  },
  methods: {
    async click(event) {
      if (event === "Delete") {
        this.isLoad = true;
        const data = await this.$store.dispatch(
          "datasets/delete",
          this.inputId
        );
        this.$store.dispatch("messages/setMessage", {
          message: `Remove dataset ${data.title}`,
        });
        this.isLoad = false;
        this.dialog = false;
        return;
      }
      if (this.$refs.form.validate()) {
        this.isLoad = true;
        const dataset = {
          id: this.inputId,
          title: this.inputName,
          size: +this.inputSize,
          tags: this.inputTags,
        };
        console.log(dataset);
        if (event === "New") {
          // const data = await this.$store.dispatch("datasets/add", dataset);
          // this.$store.dispatch("messages/setMessage", {
          // message: `Add dataset ${data.title}`,
          // });
        }
        if (event === "Edit") {
          // const data = await this.$store.dispatch("datasets/edit", dataset);
          // this.$store.dispatch("messages/setMessage", {
          // message: `Edit dataset ${data.title}`,
          // });
        }
        this.isLoad = false;
        this.dialog = false;
      }
    },
    cancel() {
      this.dialog = false;
    },
    change({ event, id }) {
      console.log(event, id);
      this.dialogTitle = event;
      const [dataset] = this.datasets.filter((item) => {
        return item.id === id;
      });
      console.log(dataset);
      this.inputId = id;
      this.inputName = dataset ? dataset.title || "" : "";
      this.inputSize = dataset ? dataset.size || 0 : 0;
      // this.inputTags = dataset ? dataset.tags || {} : {};
      this.dialog = true;
    },
  },
  watch: {
    search() {
      // Items have already been loaded
      if (this.items.length > 0) return;
      // Items have already been requested
      if (this.isLoading) return;

      this.isLoading = true;

      // Lazily load input items
      fetch("/api/v1/datasets-sources/?term=")
        .then((res) => res.json())
        .then((res) => {
          console.log(res);
          // const { count, entries } = res;
          // this.count = count;
          // this.entries = entries;
        })
        .catch((err) => {
          console.log(err);
        })
        .finally(() => (this.isLoading = false));
    },
  },
};
</script>
