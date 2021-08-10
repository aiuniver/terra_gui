<template>
  <div class="params">
    <div class="params__btn" @click="openFull">
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
          <TButton @click.native="download" :loading="loading" />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import DatasetTab from '@/components/datasets/params/DatasetTab.vue';
import DatasetButton from './DatasetButton.vue';
import TButton from '@/components/forms/Button.vue';
export default {
  name: 'Settings',
  components: {
    DatasetTab,
    DatasetButton,
    TButton,
  },
  data: () => ({
    loading: false,
    dataset: {},
    interval: null,
    inputs: 1,
    outputs: 1,
    rules: {
      length: len => v => (v || '').length >= len || `Length < ${len}`,
      required: len => len.length !== 0 || `Not be empty`,
    },
  }),
  computed: {
    ...mapGetters({
      settings: 'datasets/getSettings',
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
        this.$store.dispatch('datasets/setFull', val);
      },
      get() {
        return this.$store.getters['datasets/getFull'];
      },
    },
  },
  methods: {
    async createInterval() {
      this.interval = setTimeout(async () => {
        const data = await this.$store.dispatch('datasets/loadProgress', {});
        const {
          finished,
          message,
          percent,
          data: { file_manager },
        } = data;
        if (!data || finished) {
          // clearTimeout(this.interval);
          this.$store.dispatch('messages/setProgressMessage', message);
          this.$store.dispatch('messages/setProgress', percent);
          if (file_manager) {
            this.$store.dispatch('datasets/setFilesSource', file_manager);
          }
          this.loading = false;
          this.full = true;
        } else {
          this.$store.dispatch('messages/setProgress', percent);
          this.$store.dispatch('messages/setProgressMessage', message);
          this.createInterval();
        }
        console.log(data);
      }, 1000);
    },
    select(select) {
      console.log(select);
      this.dataset = select;
    },
    openFull() {
      if (this.$store.state.datasets.filesSource.length) {
        this.full = true;
      } else {
        this.$Modal.alert({
          width: 250,
          title: 'Внимание!',
          content: 'Загрузите датасет',
        });
      }
    },
    async download() {
      console.log('dfdfdfdfdfdf');
      const { mode, value } = this.dataset;
      if (mode && value) {
        this.loading = true;
        const data = await this.$store.dispatch('datasets/sourceLoad', { mode, value });
        console.log(data);
        if (data) {
          this.createInterval();
        }
      } else {
        this.$store.dispatch('messages/setMessage', {
          error: 'Выберите файл',
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
    border: 1px solid #a7bed3;
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
      background-image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTgiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxOCAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEgMTJIMTJDMTIuNTUgMTIgMTMgMTEuNTUgMTMgMTFDMTMgMTAuNDUgMTIuNTUgMTAgMTIgMTBIMUMwLjQ1IDEwIDAgMTAuNDUgMCAxMUMwIDExLjU1IDAuNDUgMTIgMSAxMlpNMSA3SDlDOS41NSA3IDEwIDYuNTUgMTAgNkMxMCA1LjQ1IDkuNTUgNSA5IDVIMUMwLjQ1IDUgMCA1LjQ1IDAgNkMwIDYuNTUgMC40NSA3IDEgN1pNMCAxQzAgMS41NSAwLjQ1IDIgMSAySDEyQzEyLjU1IDIgMTMgMS41NSAxMyAxQzEzIDAuNDUgMTIuNTUgMCAxMiAwSDFDMC40NSAwIDAgMC40NSAwIDFaTTE3LjMgOC44OEwxNC40MiA2TDE3LjMgMy4xMkMxNy42OSAyLjczIDE3LjY5IDIuMSAxNy4zIDEuNzFDMTYuOTEgMS4zMiAxNi4yOCAxLjMyIDE1Ljg5IDEuNzFMMTIuMyA1LjNDMTEuOTEgNS42OSAxMS45MSA2LjMyIDEyLjMgNi43MUwxNS44OSAxMC4zQzE2LjI4IDEwLjY5IDE2LjkxIDEwLjY5IDE3LjMgMTAuM0MxNy42OCA5LjkxIDE3LjY5IDkuMjcgMTcuMyA4Ljg4WiIgZmlsbD0iI0E3QkVEMyIvPgo8L3N2Zz4K);
    }
  }
  &__items {
    &--btn {
      float: right;
      width: 150px;
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
</style>