<template>
  <div class="params-full">
    <div class="params-full__inner">
      <div class="params-full__btn" @click="full = !full">
        <i class="params-full__btn--icon"></i>
      </div>
      <div :class="['params-full__files', { toggle: !toggle }]">
        <BlockFiles @toggle="change" />
      </div>
      <scrollbar class="params-full__scroll" :ops="ops">
        <div class="params-full__main">
          <div class="main__header">
            <BlockHeader />
          </div>
          <div v-if="isTable" class="main__handlers">
            <BlockHandlers />
          </div>
          <div class="main__center" :style="height">
            <div class="main__center--left">
              <BlockMainLeft />
            </div>
            <div class="main__center--right">
              <BlockMainRight />
            </div>
          </div>
          <div class="main__footer">
            <BlockFooter @create="createObject" />
          </div>
        </div>
      </scrollbar>
    </div>
  </div>
</template>

<script>
import BlockFiles from './block/BlockFiles.vue';
import BlockFooter from './block/BlockFooter.vue';
import BlockHeader from './block/BlockHeader.vue';
import BlockMainLeft from './block/BlockMainLeft.vue';
import BlockMainRight from './block/BlockMainRight.vue';
import BlockHandlers from './block/BlockHandlers.vue';
import { debounce } from '@/utils/core/utils';
export default {
  name: 'ParamsFull',
  components: {
    BlockFiles,
    BlockFooter,
    BlockHeader,
    BlockMainLeft,
    BlockMainRight,
    BlockHandlers,
  },
  data: () => ({
    toggle: true,
    debounce: null,
    ops: {
      scrollPanel: {
        scrollingX: false,
        scrollingY: true,
      },
    },
  }),
  computed: {
    // ...mapGetters({
    //   settings: "datasets/getSettings",
    // }),
    full: {
      set(val) {
        this.$store.dispatch('datasets/setFull', val);
      },
      get() {
        return this.$store.getters['datasets/getFull'];
      },
    },
    height() {
      let height = this.$store.getters['settings/height']({ style: false, clean: true });
      height = height - 172 - 96;
      // console.log(height);
      return { flex: '0 0 ' + height + 'px', height: height + 'px' };
    },
    isTable() {
      return this.$store.getters['datasets/getFilesDrop'].some(val => val.type === 'table');
    },
  },
  methods: {
    async createObject(obj) {
      this.$store.dispatch('messages/setMessage', { info: `Создается датасет "${obj.name}"` });
      const res = await this.$store.dispatch('datasets/createDataset', obj);
      console.log(res);
      if (res) {
        const { success } = res;
        if (success) {
          this.debounce(true);
        }
      }
    },
    async progress() {
      const res = await this.$store.dispatch('datasets/createProgress', {});
      if (res) {
        const { finished } = res.data;
        if (!finished) {
          this.debounce(true);
        } else {
          this.full = false;
        }
      }
    },
    change(value) {
      this.toggle = value;
    },
  },
  created() {
    this.debounce = debounce(status => {
      console.log(status);
      if (status) {
        this.progress();
      }
    }, 1000);

    // this.debounce(this.isLearning);
  },
  beforeDestroy() {
    this.debounce(false);
  },
};
</script>

<style lang="scss">
.params-full {
  flex-shrink: 0;
  width: 100%;
  padding-left: 41px;
  flex: auto;
  display: -webkit-flex;
  display: flex;
  background-color: #0e1621;
  border-top: #0e1621 solid 1px;
  &__inner {
    width: 100%;
    display: -webkit-flex;
    display: flex;
    position: relative;
    background-color: #17212b;
  }
  &__btn {
    position: absolute;
    bottom: 0px;
    left: -32px;
    width: 32px;
    height: 40px;
    background-color: #17212b;
    border-radius: 4px 0px 0px 4px;
    padding: 12px 7px;
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
  &__files {
    flex: 0 0 190px;
    display: -webkit-flex;
    display: flex;
    border-right: #0e1621 solid 1px;
    &.toggle {
      flex: 0 0 24px;
    }
  }
  &__main {
    flex: 1 1;
    display: -webkit-flex;
    display: flex;
    flex-direction: column;
    width: 100%;
    overflow: hidden;
    padding: 0 10px 0 0;
    & .main__header {
      flex: 0 0 172px;
      display: flex;
      // border-bottom: #0e1621 solid 1px;
    }
    & .main__center {
      flex: 0 0;
      display: flex;
      &--left {
        flex: 1 1;
        border-right: #0e1621 solid 1px;
        overflow: hidden;
      }
      &--right {
        flex: 1 1;
        overflow: hidden;
      }
    }
    & .main__footer {
      flex: 0 0 96px;
      height: 96px;
      overflow: hidden;
      border-top: #0e1621 solid 1px;
    }
  }
}
</style>
