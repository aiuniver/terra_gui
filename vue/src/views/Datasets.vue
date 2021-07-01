<template>
  <div>
    <Filters />
    <Dataset />
    <Params />
    <Footer />
  </div>
</template>

<script>
import Filters from '@/components/datasets/Filters.vue'
import Dataset from '@/components/datasets/Dataset.vue'
import Params from '@/components/datasets/Params.vue'
import Footer from '@/components/app/Footer.vue'

import { mapGetters } from "vuex";
// import Card from "@/components/dataset/Card";
// import Settings from "@/components/dataset/Settings";

export default {
  name: "Datasets",
  components: {
    Filters,
    Dataset,
    Params,
    Footer,
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
