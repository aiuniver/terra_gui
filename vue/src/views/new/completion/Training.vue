<template>
  <main class="page-training">
    <div class="cont">
      <Toolbar :collabse="collabse" />
      <Graphics @collabse="collabse = $event" />
      <Params />
    </div>
  </main>
</template>

<script>
// import { mapGetters } from "vuex";
import Toolbar from '@/components/training/Toolbar.vue';
import Graphics from '@/components/training/Graphics.vue';
import Params from '@/components/new/training/Params.vue';

export default {
  name: 'Training',
  components: {
    Toolbar,
    Graphics,
    Params,
  },
  data: () => ({
    collabse: [],
  }),
  methods: {},
  created() {
    // console.log(this.$store.dispatch('trainings/data'))
  },
};
</script>

<style lang="scss" scoped>
.cont {
  background: #17212b;
  padding: 0;
  display: flex;
  height: 100%;
}
</style>
