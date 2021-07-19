<template>
  <main class="page-modeling">
    <div class="container">
      <Toolbar />
      <ModCanvas2 />
      <Params />
      <ModalLoadModel />
    </div>
  </main>
</template>

<script>
import Toolbar from "@/components/modeling/Toolbar";
import ModCanvas2 from "@/components/modeling/ModCanvas2";
import Params from "@/components/modeling/Params";
import ModalLoadModel from "@/components/modeling/ModalLoadModel";

export default {
  name: "Modeling",
  components: {
    Toolbar,
    ModCanvas2,
    Params,
    ModalLoadModel,
  },
};
</script>
