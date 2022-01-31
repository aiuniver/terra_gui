<template>
  <div class="params">
    <div class="params__body">
      <div class="params__header">Данные</div>
      <scrollbar>
        <div class="params__inner">
          <component :is="getComp.component" />
        </div>
      </scrollbar>
    </div>
    <div class="params__footer">
      <Pagination :value="value" :title="getComp.title" @next="onNext" @prev="onPrev" />
    </div>
  </div>
</template>

<script>
import { mapActions, mapGetters } from 'vuex';
import { debounce } from '@/utils/core/utils';
import Preview from './Preview';
import Settings from './Settings';
import Download from './Download';
import Helpers from './Helpers';
import Pagination from './Pagination';
export default {
  components: {
    Preview,
    Settings,
    Download,
    Pagination,
    Helpers,
  },
  data: () => ({
    value: 1,
    debounce: null,
    list: [
      { id: 1, title: 'Download', component: 'download' },
      { id: 2, title: 'Preview', component: 'Preview' },
      { id: 3, title: 'Settings', component: 'settings' },
      { id: 4, title: 'Helpers', component: 'helpers' },
    ],
  }),
  computed: {
    ...mapGetters({
      select: 'createDataset/getSelectSource',
    }),
    getComp() {
      return this.list.find(i => i.id === this.value);
    },
  },
  methods: {
    ...mapActions({
      setSourceLoad: 'createDataset/setSourceLoad',
      sourceLoadProgress: 'createDataset/sourceLoadProgress',
      setOverlay: 'settings/setOverlay',
    }),
    onNext() {
      if (this.value === 1) this.onDownload();
      if (this.value < this.list.length) this.value = this.value + 1;
    },
    onPrev() {
      if (this.value > 1) this.value = this.value - 1;
    },
    async onProgress() {
      const res = await this.sourceLoadProgress();
      if (!res?.data?.finished) {
        this.debounce(true);
      } else {
        this.setOverlay(false);
      }
    },
    async onDownload() {
      const { mode, value } = this.select;
      const success = await this.setSourceLoad({ mode, value });
      if (success) {
        this.setOverlay(true);
        this.debounce(true);
      }
    },
  },
  created() {
    this.debounce = debounce(status => {
      if (status) {
        this.onProgress();
      }
    }, 1000);
  },
  beforeDestroy() {
    this.debounce(false);
  },
};
</script>

<style lang="scss">
.params {
  position: relative;
  display: flex;
  flex-direction: column;
  height: 100%;
  &__header {
    height: 50px;
    display: flex;
    align-items: center;
    border-bottom: 1px solid black;
    padding: 0 20px;
  }
  &__inner {
    padding: 0 20px;
    height: 100%;
  }
  &__body {
    flex: 1 1 auto;
    overflow: hidden;
  }
  &__footer {
    border-top: 1px solid black;
    flex: 0 0 70px;
    padding: 10px 20px;
  }
}
</style>