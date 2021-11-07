<template>
  <main class="page-deploy">
    <div class="cont">
      <Deploy />
      <Params @overlay="setOverlay" />
    </div>
    <div class="overlay" v-if="overlay"></div>
  </main>
</template>

<script>
import Params from '@/components/deploy/params/Params.vue';
import Deploy from '@/components/deploy/Deploy.vue';

export default {
  name: 'Datasets',
  components: {
    Params,
    Deploy,
  },
  data: () => ({
    overlay: false,
  }),
  methods: {
    setOverlay(value) {
      this.overlay = value;
    },
  },
};
</script>

<style lang="scss" scoped>
.page-deploy {
  height: 100%;
  width: 100%;
}
.cont {
  background: #17212b;
  padding: 0;
  display: flex;
  height: 100%;
}
.overlay {
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background-color: #0e1621;
  opacity: 0.45;
}
</style>