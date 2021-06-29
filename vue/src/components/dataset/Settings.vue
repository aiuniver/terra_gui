<template>
  <div>
    <v-tabs v-model="tab" background-color="accent" fixed-tabs>
      <v-tabs-slider color="primary"></v-tabs-slider>
      <v-tab class="success--text">
        <v-icon class="mr-2">mdi-google-drive</v-icon>
        Google Drive
      </v-tab>
      <v-tab class="success--text">
        <v-icon class="mr-2">mdi-link</v-icon>
        URL-link
      </v-tab>
    </v-tabs>
    <v-row class="pa-2">
      <v-col cols="12">
        <v-autocomplete
          v-model="model"
          :items="items"
          :loading="isLoading"
          :search-input.sync="search"
          dark
          color="primary"
          background-color="accent"
          hide-selected
          hide-details
          label="Load file"
          dense
          filled
          solo-inverted
          item-text="label"
          item-value="value"
          append-outer-icon="mdi-download"
          @click:append-outer="download"
          @focus="focus"
        ></v-autocomplete>
      </v-col>
      <v-col cols="6">
        <v-text-field
          v-model="inputs"
          type="number"
          label="Inputs"
          dense
          hide-details
          :rules="[rules.required]"
          outlined
        ></v-text-field>
      </v-col>
      <v-col cols="6">
        <v-text-field
          v-model="outputs"
          type="number"
          label="Outputs"
          dense
          hide-details
          :rules="[rules.required]"
          outlined
        ></v-text-field>
      </v-col>
      <v-col cols="12">
        <v-form ref="form">
          <Input :qty="+inputs" :settings="settings" />
        </v-form>
      </v-col>
      <v-col cols="12">
        <v-btn
          block
          outlined
          color="primary"
          :disabled="!+inputs || !+outputs"
          @click="click"
        >
          Сформировать
        </v-btn>
      </v-col>
    </v-row>
  </div>
</template>

<script>
import serialize from "@/assets/js/serialize";
import Input from "@/components/dataset/form/input/Input";
export default {
  name: "Settings",
  components: {
    Input,
  },
  data: () => ({
    inputs: 1,
    outputs: 1,
    settings: {},
    tab: null,
    model: "",
    items: [],
    isLoading: false,
    search: null,
    rules: {
      length: (len) => (v) => (v || "").length >= len || `Length < ${len}`,
      required: (len) => len.length !== 0 || `Not be empty`,
    },
  }),
  methods: {
    async download() {
      if (this.model && this.inputs && this.outputs) {
        this.isLoading = true
        const res = {
          method: "post",
          url: "/api/v1/exchange/load_dataset/",
          data: {
            link: "",
            mode: "google_drive",
            name: this.model,
            num_links: {
              inputs: +this.inputs,
              outputs: +this.outputs,
            },
          },
        };
        const data = await this.$store.dispatch("datasets/axios", res);
        this.settings = data.data;
        this.$store.dispatch("messages/setMessage", {
          message: "File is loading",
        });
        this.isLoading = false
        console.log(data);
      } else {
        this.$store.dispatch("messages/setMessage", {
          error: "Check file",
        });
      }
    },
    async focus() {
      if (!this.items.length) {
        const res = {
          method: "get",
          url: "/api/v1/datasets-sources/?term=",
        };
        const data = await this.$store.dispatch("datasets/axios", res);
        this.items = data.map((item) => {
          return item.value;
        });
      }
    },
    click() {
      console.log(this.$refs.form.$el);

      if (this.$refs.form.validate()) {
        console.log(
          serialize(this.$refs.form.$el, {
            hash: true,
            disabled: true,
            empty: true,
          })
        );
      } else {
        this.$store.dispatch("messages/setMessage", {
          error: "Error validate",
        });
      }
    },
  },
};
</script>
