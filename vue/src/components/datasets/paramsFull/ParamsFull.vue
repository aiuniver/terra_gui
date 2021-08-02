<template>
  <div class="params-full">
    <div class="params-full__inner">
      <div class="params-full__btn" @click="full = !full">
        <i class="params-full__btn--icon"></i>
      </div>
      <div :class="['params-full__files', {toggle: !toggle}]">
        <BlockFiles @toggle="change"/>
      </div>
      <div class="params-full__main">
        <div class="main__header">
          <BlockHeader />
        </div>
        <div class="main__center">
          <div class="main__center--left">
            <BlockMainLeft />
          </div>
          <div class="main__center--right">
            <BlockMainRight />
          </div>
        </div>
        <div class="main__footer">
          <BlockFooter />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import BlockFiles from "./block/BlockFiles.vue";
import BlockFooter from "./block/BlockFooter.vue";
import BlockHeader from "./block/BlockHeader.vue";
import BlockMainLeft from "./block/BlockMainLeft.vue";
import BlockMainRight from "./block/BlockMainRight.vue";

export default {
  name: "ParamsFull",
  components: {
    BlockFiles,
    BlockFooter,
    BlockHeader,
    BlockMainLeft,
    BlockMainRight,
  },
  data: () => ({
    toggle: true
  }),
  computed: {
    // ...mapGetters({
    //   settings: "datasets/getSettings",
    // }),
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
    change(value){
      this.toggle = value
    }
  }
};
</script>


<style lang="scss">
.params-full__inner {
  width: 100%;
  display: -webkit-flex;
  display: flex;
  position: relative;
}
.params-full {
  flex-shrink: 0;
  width: 100%;
  margin-left: 41px;
  flex: auto;
  display: -webkit-flex;
  display: flex;
  background-color: #17212b;
  border-top: #0e1621 solid 1px;
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
    &.toggle{
      flex: 0 0 24px;
    }
  }
  &__main {
    flex: 1 1 360px;
    display: -webkit-flex;
    display: flex;
    flex-direction: column;
    & .main__header {
      flex: 0 0 170px;
      border-bottom: #0e1621 solid 1px;
    }
    & .main__center {
      flex: 1 1;
      display: flex;
      &--left {
        flex: 1 1;
        border-right: #0e1621 solid 1px;
        width: 100%;
        height: 100%;
      }
      &--right {
        flex: 1 1;
      }
    }
    & .main__footer {
      flex: 0 0 96px;
      border-top: #0e1621 solid 1px;
    }
  }
}
</style>