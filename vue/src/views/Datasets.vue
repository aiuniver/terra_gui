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
                    v-model="inputTitle"
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
                    :items="tags"
                    :menu-props="{ maxHeight: '400' }"
                    label="Tags"
                    multiple
                    prepend-icon="mdi-tag"
                    :rules="[rules.required]"
                    persistent-hint
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
      <v-col cols="12" class="align-center pb-0 d-flex">
        <h5 color="secondary" text-color="success">Tags</h5>
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
            small
            filter
            outlined
          >
            {{ tag }}
          </v-chip>
        </v-chip-group>
      </v-col>
      <v-col cols="12" class="align-center pb-0 d-flex">
        <h5>Datasets</h5>
        <v-btn
          class="ml-2"
          elevation="2"
          small
          icon
          @click="change({ event: 'New' })"
        >
          <v-icon>mdi-plus</v-icon>
        </v-btn>
        <v-btn
          class="ml-2"
          elevation="2"
          small
          icon
        >
          <v-icon>mdi-google-drive</v-icon>
        </v-btn>
        <v-btn
          class="ml-2"
          elevation="2"
          small
          icon
        >
          <v-icon>mdi-link</v-icon>
        </v-btn>
      </v-col>
      <v-col cols="12" class="d-flex flex-wrap">
        <Card
          v-for="(dataset, i) in datasets"
          :key="i"
          :dataset="dataset"
          @change="change"
          />
        
      </v-col>
      <v-col cols="12">
        <h3 v-if="!datasets.length" class="text-center">Not datasets</h3>
      </v-col>
    </v-row>
  </div>
</template>

<script>
import { mapGetters } from "vuex";
import Card from "@/components/dataset/Card";

export default {
  name: "Datasets",
  components: {
    Card,
  },
  data: () => ({
    modalDel: false,
    dialog: false,
    dialogTitle: "",
    inputTitle: "",
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
          title: this.inputTitle,
          size: +this.inputSize,
          tags: this.inputTags,
        };
        if (event === "New") {
          const data = await this.$store.dispatch("datasets/add", dataset);
          this.$store.dispatch("messages/setMessage", {
            message: `Add dataset ${data.title}`,
          });
        }
        if (event === "Edit") {
          const data = await this.$store.dispatch("datasets/edit", dataset);
          this.$store.dispatch("messages/setMessage", {
            message: `Edit dataset ${data.title}`,
          });
        }
        this.isLoad = false;
        this.dialog = false;
      }
    },
    cancel() {
      this.dialog = false;
    },
    change({ event, id }) {
      this.dialogTitle = event;
      const [dataset] = this.datasets.filter((item) => {
        return item.id === id;
      });
      this.inputId = id;
      this.inputTitle = dataset ? dataset.title || "" : "";
      this.inputSize = dataset ? dataset.size || 0 : 0;
      this.inputTags = dataset ? dataset.tags || [] : [];
      this.dialog = true;
    },
  },
};
</script>
