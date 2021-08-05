<template>
  <div class="params">
    <div class="params__btn" @click="full = !full">
      <i class="params__btn--icon"></i>
    </div>
    <div class="params__items">
      <div class="params__items--item">
        <DatasetButton />
      </div>
      <div class="params__items--item pa-0">
        <DatasetTab @select="select" />
      </div>
      <div class="params__items--item">
        <div class="params__items--btn">
          <button @click="download" v-text="'Загрузить'"/>   
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters } from "vuex";
import DatasetTab from "@/components/datasets/params/DatasetTab.vue";
import DatasetButton from "./DatasetButton.vue";
export default {
  name: "Settings",
  components: {
    DatasetTab,
    DatasetButton,
  },
  data: () => ({
    dataset: {},
    interval: null,
    inputs: 1,
    outputs: 1,
    rules: {
      length: (len) => (v) => (v || "").length >= len || `Length < ${len}`,
      required: (len) => len.length !== 0 || `Not be empty`,
    },
  }),
  computed: {
    ...mapGetters({
      settings: "datasets/getSettings",
    }),
    inputLayer() {
      const int = +this.inputs;
      const settings = this.settings;
      return int > 0 && int < 100 && Object.keys(settings).length ? int : 0;
    },
    outputLayer() {
      const int = +this.outputs;
      const settings = this.settings;
      return int > 0 && int < 100 && Object.keys(settings).length ? int : 0;
    },
    full: {
      set(val) {
        this.$store.dispatch("datasets/setFull", val);
      },
      get() {
        return this.$store.getters["datasets/getFull"];
      },
    },
  },
  methods: {
    createInterval() {
      this.interval = setInterval(async () => {
        const data = await this.$store.dispatch("datasets/loadProgress", {});
        const { finished, message, percent } = data;
        if (!data || finished) {
          clearTimeout(this.interval);
          this.$store.dispatch("messages/setProgressMessage", message);
          this.$store.dispatch("messages/setProgress", percent);
        } else {
          this.$store.dispatch("messages/setProgress", percent);
          this.$store.dispatch("messages/setProgressMessage", message);
        }
        console.log(data);
      }, 1000);
    },
    select(select) {
      console.log(select);
      this.dataset = select;
    },
    async download() {
      const { mode, value } = this.dataset;
      if (mode && value) {
        this.createInterval();
        await this.$store.dispatch("datasets/sourceLoad", { mode, value });
      } else {
        this.$store.dispatch("messages/setMessage", {
          error: "Выберите файл",
        });
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.params {
  width: 400px;
  flex-shrink: 0;
  border-left: #0e1621 solid 1px;
  background-color: #17212b;
  position: relative;
  // border-left: #0e1621  1px solid;
  &__btn {
    position: absolute;
    bottom: 1px;
    right: 0px;
    width: 31px;
    height: 38px;
    background-color: #17212b;
    border-radius: 4px 0px 0px 4px;
    border: 1px solid #A7BED3;
    padding: 10px 7px 12px 7px;
    cursor: pointer;
    &--icon {
      display: block;
      width: 17px;
      height: 15px;
      background-position: center;
      background-repeat: no-repeat;
      -webkit-user-select: none;
      -moz-user-select: none;
      -ms-user-select: none;
      user-select: none;
      background-image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTgiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxOCAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTE3IDEySDZDNS40NSAxMiA1IDExLjU1IDUgMTFDNSAxMC40NSA1LjQ1IDEwIDYgMTBIMTdDMTcuNTUgMTAgMTggMTAuNDUgMTggMTFDMTggMTEuNTUgMTcuNTUgMTIgMTcgMTJaTTE3IDdIOUM4LjQ1IDcgOCA2LjU1IDggNkM4IDUuNDUgOC40NSA1IDkgNUgxN0MxNy41NSA1IDE4IDUuNDUgMTggNkMxOCA2LjU1IDE3LjU1IDcgMTcgN1pNMTggMUMxOCAxLjU1IDE3LjU1IDIgMTcgMkg2QzUuNDUgMiA1IDEuNTUgNSAxQzUgMC40NSA1LjQ1IDAgNiAwSDE3QzE3LjU1IDAgMTggMC40NSAxOCAxWk0wLjcwMDAwMSA4Ljg4TDMuNTggNkwwLjcwMDAwMSAzLjEyQzAuMzEwMDAxIDIuNzMgMC4zMTAwMDEgMi4xIDAuNzAwMDAxIDEuNzFDMS4wOSAxLjMyIDEuNzIgMS4zMiAyLjExIDEuNzFMNS43IDUuM0M2LjA5IDUuNjkgNi4wOSA2LjMyIDUuNyA2LjcxTDIuMTEgMTAuM0MxLjcyIDEwLjY5IDEuMDkgMTAuNjkgMC43MDAwMDEgMTAuM0MwLjMyMDAwMiA5LjkxIDAuMzEwMDAxIDkuMjcgMC43MDAwMDEgOC44OFoiIGZpbGw9IiNBN0JFRDMiLz4KPC9zdmc+Cg==);
    }
  }
  &__items {
    &--btn{
      button{
        width: 100px;
        float: right;
      }
    }
    &--item {
      padding: 20px;
    }
    &--title {
      display: block;
      line-height: 1.25;
      margin: 0 0 10px 0;
      padding: 5px 20px;
      font-size: 0.75rem;
      user-select: none;
      background-color: #0e1621;
    }
  }
}
button {
  font-size: 0.875rem;
}
</style>