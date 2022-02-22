<template>
  <div class="params">
    <div class="params__body">
      <div class="params__header mb-2">{{ getComp.title }}</div>
      <scrollbar>
        <div class="params__inner">
          <component :is="getComp.component" :state="value" />
        </div>
      </scrollbar>
    </div>
    <div class="params__footer">
      <Pagination :value="value" :list="list" @next="onNext" @prev="onPrev" @create="onCreate" />
    </div>
  </div>
</template>

<script>
import { isError } from '@/store/const/create';
import { mapActions } from 'vuex';
import { debounce } from '@/utils/core/utils';
import StateThree from './settings/';
import StateOne from './StateOne';
import StateFour from './StateFour';
import Pagination from './Pagination';
export default {
  components: {
    StateOne,
    StateThree,
    Pagination,
    StateFour,
  },
  props: {
    pagination: {
      type: Number,
      default: 1,
    },
  },
  data: () => ({
    debounce: null,
    list: [
      { id: 1, title: 'Данные', component: 'state-one' },
      { id: 2, title: 'Input', component: 'state-three' },
      { id: 3, title: 'Output', component: 'state-three' },
      { id: 4, title: 'Завершение', component: 'state-four' },
    ],
  }),
  computed: {
    getComp() {
      return this.list.find(i => i.id === this.value);
    },
    value: {
      set(value) {
        this.setPagination(value);
      },
      get() {
        return this.pagination;
      },
    },
  },
  methods: {
    ...mapActions({
      setSourceLoad: 'createDataset/setSourceLoad',
      sourceLoadProgress: 'createDataset/sourceLoadProgress',
      setPagination: 'createDataset/setPagination',
      create: 'createDataset/create',
      datasetValidate: 'createDataset/datasetValidate',
      setOverlay: 'settings/setOverlay',
      blockSelect: 'create/main',
    }),
    async onValidate(type) {
      return await this.datasetValidate(type);
    },
    async onNext() {
      let errors = {};
      if (this.value === 1) await this.onDownload();

      if (this.value === 2) {
        errors = await this.onValidate('inputs') ;
      }
      if (this.value === 3) {
        errors = await this.onValidate('outputs');
      }
      if (!isError(errors) && this.value < this.list.length) this.value = this.value + 1;
    },
    onPrev() {
      if (this.value > 1) this.value = this.value - 1;
    },
    onCreate() {
      console.log('create');
      this.create();
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
      const success = await this.setSourceLoad();
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
  watch: {
    value(value, old) {
      console.log(value, old);
      this.blockSelect({ value, old });
    },
  },
};
</script>

<style lang="scss">
.params {
  user-select: none;
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