<template>
  <main class="page-deploy">
    <div class="cont">
      <Deploy />
      <Params />
    </div>
  </main>
</template>

<script>
import Params from "@/components/deploy/params/Params.vue";
import Deploy from "@/components/deploy/Deploy.vue";

export default {
  name: "Datasets",
  components: {
    Params,
    Deploy
  },
};
</script>

<style lang="scss" scoped>
.cont{
  background: #17212B;
  padding: 0;
  display: flex;
}

</style>