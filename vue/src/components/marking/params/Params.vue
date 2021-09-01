<template>
  <div class="params">
    <div class="params__items">
      <div class="params__items--item pa-0">
        <DatasetTab v-model="tab" />
      </div>
      <div class="params__items--item">
        <div class="params__items--btn">
          <t-button :loading="loading" :disabled="disabled" />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import DatasetTab from '@/components/datasets/params/DatasetTab.vue';
export default {
  name: 'Settings',
  components: {
    DatasetTab,
  },
  data: () => ({
    tab: 'GoogleDrive',
    loading: false,
    dataset: {},
    prevSet: '',
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
    disabled() {
      if (Object.keys(this.dataset).length === 0 && this.dataset.mode === 'GoogleDrive') {
        return true;
      } else if (!this.dataset.value?.value && this.dataset.mode === 'URL') {
        return true;
      } else {
        return this.tab !== this.dataset.mode;
      }
    },
  },
  methods: {
    // saveSet() {
    //   if (this.dataset.mode === 'GoogleDrive') {
    //     this.prevSet = this.dataset
    //     this.$el.querySelector('.t-field__input').value = ''
    //   }
    //   if (this.dataset.mode === 'URL') this.dataset = this.prevSet
    // },
    // select(select) {
    //   this.dataset = select;
    // },
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
      width: 100%;
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